#include <I2S.h>
#include <arduinoFFT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include <Arduino.h>
#include <math.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Audio Configuration
const uint16_t sampleRate = 16000;
const uint16_t samples = 128;
const uint16_t melBands = 64;
const uint16_t spectrogramWidth = 126;

// Hierarchical Classification (264 species, 20 genera, 5 families)
const uint16_t numFamilies = 5;
const uint16_t numGenera = 20;
const uint16_t numSpecies = 264;

// BYOL Configuration
const int EMBED_DIM = 64;
const int PROJ_DIM = 32;
const float MOMENTUM = 0.995f;
const float LR = 1e-3f;
const int FED_ROUND_STEPS = 5;

// WiFi Configuration
const char* ssid = "Network";
const char* password = "Excels!or";
const char* server_url = "http://192.168.0.108:5000/weights";

// BYOL Trainable Parameters
float proj_w[PROJ_DIM][EMBED_DIM];
float proj_b[PROJ_DIM];
float pred_w1[PROJ_DIM][PROJ_DIM];
float pred_b1[PROJ_DIM];
float pred_w2[PROJ_DIM][PROJ_DIM];
float pred_b2[PROJ_DIM];
float proj_target_w[PROJ_DIM][EMBED_DIM];
float proj_target_b[PROJ_DIM];

// Working Buffers
float embed_online[EMBED_DIM];
int localStep = 0;

// FFT Variables
double vReal[samples];
double vImag[samples];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, samples, sampleRate);

// Mel Spectrogram
float melFilterBank[melBands][samples/2];
float melSpectrogramBuffer[spectrogramWidth][melBands];
int spectrogramIndex = 0;

// TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output_family = nullptr;
TfLiteTensor* output_genus = nullptr;
TfLiteTensor* output_species = nullptr;
tflite::MicroErrorReporter micro_error_reporter;

constexpr int tensorArenaSize = 180 * 1024;
alignas(16) uint8_t* tensorArena = nullptr;

// Hierarchical Taxonomy
struct BirdTaxonomy {
  int species_to_genus[264];
  int species_to_family[264];
  
  void init() {
    int species_per_genus = 13;
    int genera_per_family = 4;
    
    for (int species_id = 0; species_id < numSpecies; species_id++) {
      int genus_id = species_id / species_per_genus;
      int family_id = genus_id / genera_per_family;
      species_to_genus[species_id] = min(genus_id, numGenera - 1);
      species_to_family[species_id] = min(family_id, numFamilies - 1);
    }
  }
} taxonomy;

// Matrix-vector multiplication
void matvec_mul(const float W[][EMBED_DIM], const float* x, const float* b, float* y, int out_dim) {
  for (int i = 0; i < out_dim; i++) {
    y[i] = b[i];
    for (int j = 0; j < EMBED_DIM; j++) {
      y[i] += W[i][j] * x[j];
    }
  }
}

void matvec_mul_proj(const float W[][PROJ_DIM], const float* x, const float* b, float* y, int out_dim) {
  for (int i = 0; i < out_dim; i++) {
    y[i] = b[i];
    for (int j = 0; j < PROJ_DIM; j++) {
      y[i] += W[i][j] * x[j];
    }
  }
}

void relu(float* x, int n) {
  for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

void normalize_l2(float* x, int n) {
  float norm = 1e-12;
  for (int i = 0; i < n; i++) norm += x[i] * x[i];
  norm = sqrt(norm);
  for (int i = 0; i < n; i++) x[i] /= norm;
}

void copy_array(float* dst, const float* src, int n) {
  for (int i = 0; i < n; i++) dst[i] = src[i];
}

// Update target network with EMA
void update_target_proj() {
  for (int i = 0; i < PROJ_DIM; i++) {
    for (int j = 0; j < EMBED_DIM; j++) {
      proj_target_w[i][j] = MOMENTUM * proj_target_w[i][j] + (1.0f - MOMENTUM) * proj_w[i][j];
    }
    proj_target_b[i] = MOMENTUM * proj_target_b[i] + (1.0f - MOMENTUM) * proj_b[i];
  }
}

// BYOL training step
float byol_step(const float* encoder_embed) {
  copy_array(embed_online, encoder_embed, EMBED_DIM);

  // Online projection
  float z[PROJ_DIM];
  matvec_mul(proj_w, embed_online, proj_b, z, PROJ_DIM);
  relu(z, PROJ_DIM);

  // Predictor
  float p1[PROJ_DIM], p[PROJ_DIM];
  matvec_mul_proj(pred_w1, z, pred_b1, p1, PROJ_DIM);
  relu(p1, PROJ_DIM);
  matvec_mul_proj(pred_w2, p1, pred_b2, p, PROJ_DIM);
  normalize_l2(p, PROJ_DIM);

  // Target projection
  float z_t[PROJ_DIM];
  matvec_mul(proj_target_w, embed_online, proj_target_b, z_t, PROJ_DIM);
  normalize_l2(z_t, PROJ_DIM);

  // MSE loss and gradients
  float loss = 0;
  float grad_out[PROJ_DIM];
  for (int i = 0; i < PROJ_DIM; i++) {
    grad_out[i] = 2.0f * (p[i] - z_t[i]);
    loss += (p[i] - z_t[i]) * (p[i] - z_t[i]);
  }

  // Backprop through predictor
  float grad_p1[PROJ_DIM];
  for (int j = 0; j < PROJ_DIM; j++) {
    float s = 0;
    for (int i = 0; i < PROJ_DIM; i++) s += pred_w2[i][j] * grad_out[i];
    grad_p1[j] = (p1[j] > 0) ? s : 0;
  }

  // Update predictor weights
  for (int i = 0; i < PROJ_DIM; i++) {
    for (int j = 0; j < PROJ_DIM; j++) {
      pred_w2[i][j] -= LR * grad_out[i] * p1[j];
      pred_w1[i][j] -= LR * grad_p1[i] * z[j];
    }
    pred_b2[i] -= LR * grad_out[i];
    pred_b1[i] -= LR * grad_p1[i];
  }

  // Backprop to projection
  float grad_z[PROJ_DIM];
  for (int j = 0; j < PROJ_DIM; j++) {
    float s = 0;
    for (int i = 0; i < PROJ_DIM; i++) s += pred_w1[i][j] * grad_p1[i];
    grad_z[j] = s;
  }

  // Update projection weights
  for (int i = 0; i < PROJ_DIM; i++) {
    for (int j = 0; j < EMBED_DIM; j++) {
      proj_w[i][j] -= LR * grad_z[i] * embed_online[j];
    }
    proj_b[i] -= LR * grad_z[i];
  }

  update_target_proj();
  return loss;
}

// Initialize BYOL parameters
void init_byol_head() {
  for (int i = 0; i < PROJ_DIM; i++) {
    proj_b[i] = 0.1f;
    proj_target_b[i] = 0.1f;
    pred_b1[i] = 0.1f;
    pred_b2[i] = 0.1f;
    
    for (int j = 0; j < EMBED_DIM; j++) {
      proj_w[i][j] = ((float)random(-50, 50) / 1000.0f);
      proj_target_w[i][j] = proj_w[i][j];
    }
    
    for (int j = 0; j < PROJ_DIM; j++) {
      pred_w1[i][j] = ((float)random(-50, 50) / 1000.0f);
      pred_w2[i][j] = ((float)random(-50, 50) / 1000.0f);
    }
  }
}

// Serialize weights for federated learning
void serialize_weights(float* buf) {
  int idx = 0;
  for (int i = 0; i < PROJ_DIM; i++) {
    buf[idx++] = proj_b[i];
    for (int j = 0; j < EMBED_DIM; j++) buf[idx++] = proj_w[i][j];
  }
  for (int i = 0; i < PROJ_DIM; i++) {
    buf[idx++] = pred_b1[i];
    buf[idx++] = pred_b2[i];
    for (int j = 0; j < PROJ_DIM; j++) {
      buf[idx++] = pred_w1[i][j];
      buf[idx++] = pred_w2[i][j];
    }
  }
}

// Load weights from server
void load_weights_from_server(const float* buf) {
  int idx = 0;
  for (int i = 0; i < PROJ_DIM; i++) {
    proj_b[i] = buf[idx++];
    for (int j = 0; j < EMBED_DIM; j++) proj_w[i][j] = buf[idx++];
  }
  for (int i = 0; i < PROJ_DIM; i++) {
    pred_b1[i] = buf[idx++];
    pred_b2[i] = buf[idx++];
    for (int j = 0; j < PROJ_DIM; j++) {
      pred_w1[i][j] = buf[idx++];
      pred_w2[i][j] = buf[idx++];
    }
  }
}

// Send weights to server
void send_weights_to_server() {
  int total_size = PROJ_DIM * (EMBED_DIM + 1) + 2 * PROJ_DIM * (PROJ_DIM + 1);
  float* buffer = new float[total_size];
  serialize_weights(buffer);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(server_url);
    http.addHeader("Content-Type", "application/octet-stream");
    http.POST((uint8_t*)buffer, total_size * sizeof(float));
    http.end();
    Serial.println("Weights sent to server");
  }
  delete[] buffer;
}

// Get global weights from server
void get_global_weights_from_server() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(server_url);
    int httpCode = http.GET();
    if (httpCode == HTTP_CODE_OK) {
      int len = http.getSize();
      uint8_t* payload = new uint8_t[len];
      WiFiClient* stream = http.getStreamPtr();
      stream->readBytes(payload, len);
      load_weights_from_server((float*)payload);
      delete[] payload;
      Serial.println("Global weights received");
    }
    http.end();
  }
}

// Mel scale conversions
float freqToMel(float freq) {
  return 2595.0 * log10(1.0 + freq / 700.0);
}

float melToFreq(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Initialize mel filterbank
void initMelFilterBank() {
  float melMin = freqToMel(0);
  float melMax = freqToMel(sampleRate / 2.0);
  float melStep = (melMax - melMin) / (melBands + 1);
  
  float melPoints[melBands + 2];
  for (int i = 0; i < melBands + 2; i++) {
    melPoints[i] = melToFreq(melMin + i * melStep);
  }
  
  for (int i = 0; i < melBands; i++) {
    for (int j = 0; j < samples / 2; j++) {
      float freq = (float)j * sampleRate / samples;
      if (freq >= melPoints[i] && freq <= melPoints[i + 1]) {
        melFilterBank[i][j] = (freq - melPoints[i]) / (melPoints[i + 1] - melPoints[i]);
      } else if (freq >= melPoints[i + 1] && freq <= melPoints[i + 2]) {
        melFilterBank[i][j] = (melPoints[i + 2] - freq) / (melPoints[i + 2] - melPoints[i + 1]);
      } else {
        melFilterBank[i][j] = 0.0;
      }
    }
  }
}

// Compute mel frame
void computeMelFrame(float* melFrame) {
  FFT.windowing(vReal, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, samples, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, samples);
  
  for (int i = 0; i < melBands; i++) {
    melFrame[i] = 0;
    for (int j = 0; j < samples / 2; j++) {
      melFrame[i] += vReal[j] * melFilterBank[i][j];
    }
    melFrame[i] = 20.0 * log10(melFrame[i] + 1e-10);
  }
}

// Prepare input tensor for hierarchical model (1x64x126)
void prepareInputTensor() {
  float* inputData = input->data.f;
  if (inputData == nullptr) return;
  
  // Find min/max for normalization
  float minVal = melSpectrogramBuffer[0][0];
  float maxVal = melSpectrogramBuffer[0][0];
  for (int i = 0; i < spectrogramWidth; i++) {
    for (int j = 0; j < melBands; j++) {
      if (melSpectrogramBuffer[i][j] < minVal) minVal = melSpectrogramBuffer[i][j];
      if (melSpectrogramBuffer[i][j] > maxVal) maxVal = melSpectrogramBuffer[i][j];
    }
  }
  
  // Fill input tensor (1, 64, 126) - channel first format
  for (int h = 0; h < melBands; h++) {
    for (int w = 0; w < spectrogramWidth; w++) {
      float normalized = (melSpectrogramBuffer[w][h] - minVal) / (maxVal - minVal + 1e-10);
      inputData[h * spectrogramWidth + w] = normalized;
    }
  }
}

// Run hierarchical inference
void runInference() {
  prepareInputTensor();
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return;
  }
  
  // Get hierarchical outputs
  float* family_out = output_family->data.f;
  float* genus_out = output_genus->data.f;
  float* species_out = output_species->data.f;
  
  // Find top prediction for each level
  int pred_family = 0, pred_genus = 0, pred_species = 0;
  float max_family = family_out[0], max_genus = genus_out[0], max_species = species_out[0];
  
  for (int i = 1; i < numFamilies; i++) {
    if (family_out[i] > max_family) {
      max_family = family_out[i];
      pred_family = i;
    }
  }
  
  for (int i = 1; i < numGenera; i++) {
    if (genus_out[i] > max_genus) {
      max_genus = genus_out[i];
      pred_genus = i;
    }
  }
  
  for (int i = 1; i < numSpecies; i++) {
    if (species_out[i] > max_species) {
      max_species = species_out[i];
      pred_species = i;
    }
  }
  
  // Softmax for confidence
  float family_conf = exp(max_family) / (exp(max_family) + 1.0);
  float species_conf = exp(max_species);
  float species_sum = 0;
  for (int i = 0; i < numSpecies; i++) species_sum += exp(species_out[i]);
  species_conf = species_conf / species_sum * 100.0;
  
  Serial.println("\n=== Hierarchical Prediction ===");
  Serial.printf("Family: %d (conf: %.1f%%)\n", pred_family, family_conf * 100);
  Serial.printf("Genus: %d\n", pred_genus);
  Serial.printf("Species: %d (conf: %.1f%%)\n", pred_species, species_conf);
  Serial.println("===============================\n");
}

// Initialize TensorFlow Lite
void initTFLite() {
  tensorArena = (uint8_t*)heap_caps_malloc(tensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tensorArena == nullptr) {
    tensorArena = (uint8_t*)malloc(tensorArenaSize);
    if (tensorArena == nullptr) {
      Serial.println("ERROR: Failed to allocate tensor arena!");
      while (1) delay(1000);
    }
  }
  
  Serial.printf("Tensor arena: %d KB\n", tensorArenaSize / 1024);
  
  model = tflite::GetModel(g_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model version mismatch!\n");
    while (1) delay(1000);
  }
  
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddMean();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddQuantize();
  resolver.AddDequantize();
  
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensorArena, tensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    while (1) delay(1000);
  }
  
  // Get input and 3 hierarchical outputs
  input = interpreter->input(0);

  Serial.println(interpreter->outputs_size());
  output_family = interpreter->output(0);
  output_genus = interpreter->output(1);
  output_species = interpreter->output(2);
  
  if (!input || !output_family || !output_genus || !output_species) {
    Serial.println("ERROR: Failed to get tensors!");
    while (1) delay(1000);
  }
  
  Serial.printf("Input: [%d, %d, %d]\n", input->dims->data[0], input->dims->data[1], input->dims->data[2]);
  Serial.printf("Outputs: Family=%d, Genus=%d, Species=%d\n", 
                output_family->dims->data[1], output_genus->dims->data[1], output_species->dims->data[1]);
  Serial.println("Model loaded successfully!");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== Hierarchical Bird Classification ===");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  
  // Initialize I2S
  I2S.setAllPins(-1, 42, 41, -1, -1);
  if (!I2S.begin(PDM_MONO_MODE, sampleRate, 16)) {
    Serial.println("ERROR: I2S init failed!");
    while (1) delay(1000);
  }
  Serial.println("I2S initialized");
  
  // Initialize mel filterbank and taxonomy
  initMelFilterBank();
  taxonomy.init();
  Serial.println("Mel filterbank and taxonomy initialized");
  
  // Initialize TensorFlow Lite
  initTFLite();
  
  // Connect WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nWiFi connected: %s\n", WiFi.localIP().toString().c_str());
  
  // Initialize BYOL
  init_byol_head();
  Serial.println("BYOL initialized");
  
  Serial.println("\n=== System Ready ===\n");
}

void loop() {
  // Collect audio and compute mel frame
  int bytesRead = I2S.read(vReal, samples * sizeof(int16_t));
  if (bytesRead > 0) {
    computeMelFrame(melSpectrogramBuffer[spectrogramIndex]);
    spectrogramIndex++;
    
    if (spectrogramIndex >= spectrogramWidth) {
      // Run inference
      prepareInputTensor();
      interpreter->Invoke();
      
      // Get encoder embedding for BYOL (assume last hidden layer before classification)
      float* encoder_embed = output_species->data.f;
      
      // BYOL training step
      // float loss = byol_step(encoder_embed);
      // Serial.printf("BYOL loss: %.5f\n", loss);
      
      // Federated learning round
      // localStep++;
      // if (localStep >= FED_ROUND_STEPS) {
      //   send_weights_to_server();
      //   delay(500);
      //   get_global_weights_from_server();
      //   localStep = 0;
      // }
      
      // Display prediction
      runInference();
      
      spectrogramIndex = 0;
    }
  }
  
  delay(10);
}