from arbiter import *
from expert import *
from train import *
from data import *
import os
from tensorflow.keras import mixed_precision

def main():
    """Main training pipeline"""
    arbiter = build_feature_extractor_arbiter_with_value(
                    mfcc_timesteps=MFCC_TIMESTEPS,
                    mfcc_features=MFCC_FEATURES,
                    latent_dim=LATENT_CHANNEL,
                    num_experts=NUM_EXPERTS
                )
    
    expert_data_base_path = '/mnt/Stuff/phd_projects/esp32-projects/bird_call_id/birds/'
    rl_data_path = '/mnt/Stuff/phd_projects/esp32-projects/bird_call_id/rl_data/'

    experts = [
        build_expert_model(LATENT_DIM) for i in range(NUM_EXPERTS)
    ]

    print(os.listdir(expert_data_base_path))
    expert_data = []

    for i in os.listdir(expert_data_base_path):
        path = f'{expert_data_base_path}/{i}'
        expert_data.append(create_bird_dataset(path))
   
    print("=== Bird Call Detection System for MCU ===")
    print(f"Total Classes: {NUM_CLASSES}")
    print(f"Number of Experts: {NUM_EXPERTS}")
    print(f"Classes per Expert: {10}")
    print(f"Latent Dimension: {LATENT_DIM}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    for expert, expert_data in zip(experts, expert_data):
        train_expert_supervised(arbiter, expert, expert_data[0], expert_data[1])
    # train_experts_supervised(system, train_data, val_data, epochs=10)

    rl_data = create_bird_dataset(rl_data_path)
    
    # Step 2: Train arbiter with RL
    train_arbiter_rl(arbiter, experts, rl_data[0])
    
    # Step 3: Knowledge distillation for compression
    # (You would create smaller student models and distill each component)
    # print("\n[Distillation step - create student models as needed]")
    
    # Step 4: Convert to TFLite
    # print("\n=== Converting Models to TFLite ===")
    
    # # Create full models for conversion
    # feature_input = keras.Input(shape=INPUT_SHAPE)
    # latent = system.feature_extractor(feature_input)
    
    # # Feature extractor model
    # feature_model = keras.Model(inputs=feature_input, outputs=latent)
    # convert_to_tflite(feature_model, "feature_extractor", quantize=True)
    
    # # Arbiter model
    # latent_input = keras.Input(shape=(LATENT_DIM,))
    # policy_logits, value = system.arbiter(latent_input)
    # arbiter_model = keras.Model(inputs=latent_input, outputs=[policy_logits, value])
    # convert_to_tflite(arbiter_model, "arbiter", quantize=True)
    
    # # Expert models
    # for expert_id, expert in system.experts.items():
    #     expert_output = expert(latent_input)
    #     expert_model = keras.Model(inputs=latent_input, outputs=expert_output)
    #     convert_to_tflite(expert_model, f"expert_{expert_id}", quantize=True)
    
    # print("\n=== Training Complete ===")
    # print("Models are ready for MCU deployment!")

main()