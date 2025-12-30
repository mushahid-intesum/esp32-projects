from arbiter import *
from expert import *
from train import *
from data import *
import os
from tensorflow.keras import mixed_precision

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    arbiter = build_feature_extractor_arbiter_with_value(
                    mfcc_timesteps=MFCC_TIMESTEPS,
                    mfcc_features=MFCC_FEATURES,
                    latent_dim=LATENT_CHANNEL,
                    num_experts=NUM_EXPERTS
                )
    
    expert_data_base_path = '/mnt/Stuff/phd_projects/esp32-projects/bird_call_identifier_training/expert_data/'
    rl_data_path = '/mnt/Stuff/phd_projects/esp32-projects/bird_call_identifier_training/arbiter_data/'

    experts = []
    expert_data = []
    expert_labels = {}

    """Expert Training"""
    # for i, folder_name in enumerate(os.listdir(expert_data_base_path)):
    #     path = f'{expert_data_base_path}/{folder_name}'
    #     expert_data.append(create_bird_dataset(path))
    #     experts.append(build_expert_model(LATENT_DIM, len(os.listdir(path))))
    #     expert_labels[i] = os.listdir(path)
    
    # expert_id = 0
    # for expert, expert_data in zip(experts, expert_data):
    #     train_expert_supervised(arbiter, expert, expert_id, expert_data[0], expert_data[1], epochs=100)
    #     expert_id += 1
    
    # for i, expert in enumerate(experts):
    #     expert.save_weights(f'./expert_{i}.weights.h5')

    """Arbiter Training"""
    for i, folder_name in enumerate(os.listdir(expert_data_base_path)):
        path = f'{expert_data_base_path}/{folder_name}'
        expert = build_expert_model(LATENT_DIM, len(os.listdir(path)))
        expert.load_weights(f'./expert_{i}.weights.h5')
        experts.append(expert)
        expert_labels[i] = os.listdir(path)

    rl_data = create_bird_dataset(rl_data_path)

    # if os.path.exists('./arbiter.weights.h5'):
    #     arbiter.load_weights('./arbiter.weights.h5')
    
    train_arbiter_rl(arbiter, experts, rl_data[0], expert_labels, episodes=50)

    arbiter.save_weights(f'./arbiter.weights.h5')
    # Step 3: Knowledge distillation for compression
    # (You would create smaller student models and distill each component)
    # print("\n[Distillation step - create student models as needed]")

    # student_arbiter = build_feature_extractor_student_arbiter_with_value(
    #                 mfcc_timesteps=MFCC_TIMESTEPS,
    #                 mfcc_features=MFCC_FEATURES,
    #                 latent_dim=LATENT_CHANNEL,
    #                 num_experts=NUM_EXPERTS
    #             )

    # student_experts = []
    # for i, folder_name in enumerate(os.listdir(expert_data_base_path)):
    #     path = f'{expert_data_base_path}/{folder_name}'
    #     expert_data.append(create_bird_dataset(path))
    #     student_experts.append(build_student_expert_model(LATENT_DIM, len(os.listdir(path))))
    
    # train_arbiter_distillation_with_features(student_arbiter, arbiter, rl_data[0])

    # for i in range(len(experts)):
    #     train_arbiter_distillation_with_features(student_experts[i], experts[i], expert_data[i][0])

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