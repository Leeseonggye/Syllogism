import argparse
from datetime import datetime

def make_parser() :
    parser = argparse.ArgumentParser(description='GPT2 Train for Syllogistic Generation')
    
    ### Wandb Setting
    parser.add_argument(
        '-project_name',
        type = str,
        help = 'Project name',
        default = 'GPT2-small Final test')
    
    parser.add_argument(
        '-project_name_middleterm',
        type = str,
        help = 'Project name',
        default = 'GPT2-small not middle term finetuning')
    
    parser.add_argument(
        '-group_name',
        type = str,
        help = 'Group name for K-Fold Validation',
        default = 'Debugging')
    
    parser.add_argument(
        '-seed',
        type=int,
        default=42)

    parser.add_argument(
        '-data_path', 
        type = str, 
        default = "/home/seonggye/Syllogism-1/Syllogistic-Commonsense-Reasoning/gpt2_practice/datasets/",
        help='학습 데이터 위치')
    
    parser.add_argument(
        '--accumulation_steps',
        type = int,
        default = 2,
        help = "gradient accumulation step")

    parser.add_argument(
        '-model_path', 
        type = str,
        default = "/home/seonggye/Syllogism-1/Syllogistic-Commonsense-Reasoning/gpt2_practice/model_log/input=p1+p2+so/",
        help = '모델 저장할 위치')
    
    parser.add_argument(
        '-model_path_middleterm', 
        type = str,
        default = "/home/seonggye/Syllogism-1/Syllogistic-Commonsense-Reasoning/gpt2_practice/model_log_middleterm",
        help = '모델 저장할 위치')

    parser.add_argument(
        '-generation_path',
        type = str,
        default = "/home/seonggye/Syllogism-1/Syllogistic-Commonsense-Reasoning/gpt2_practice/generation_log/input=p1+p2+so/",
        help = '생성된 텍스트 저장할 위치')

    parser.add_argument(
        '-num_epochs',
        type = int,
        default = 20)

    parser.add_argument(
        '-num_iteration',
        type = int,
        default = 1000000,
        help = "학습을 종료할 목표 이터레이션, epoch보다 우선시 됨")

    parser.add_argument(
        '-max_len',
        type = int,
        default = 256,
        help = "모델의 max len을 따르되 필요 시 수정")
    
    parser.add_argument(
        '-batch_size',
        type = int,
        default = 4)

    parser.add_argument(
        "-learning_rate",
        type = float,
        default = 1e-5)
    
    parser.add_argument(
        "-log_interval",
        type = int,
        default = 10000,
        help = "로그를 출력할 이터레이션 간격")

    parser.add_argument(
        "-eval_interval",
        type = int,
        default = 100,
        help = "평가를 수행할 이터레이션 간격")

    parser.add_argument(
        '-eval_batch_size',
        type = int,
        default = 16)

    parser.add_argument(
        '-save_interval',
        type = int,
        default = 1000,
        help = "모델을 저장할 이터레이션 간격")
    
    parser.add_argument(
        '-save_generation_interval',
        type = int,
        default = 1000,
        help = "생성된 텍스트를 저장할 간격")

    parser.add_argument(
        "-kfold",
        type = int,
        default = 5,
        help = "kfold 개수")
    
    parser.add_argument(
        '-save_final_model',
        type = bool,
        default = False,
        help = "최종 모델을 저장할지 여부")
    
    parser.add_argument(
        '-save_epochs',
        type = bool,
        default = False,
        help = "매 에폭마다 모델, 생성된 텍스트를 저장할지 여부")

    parser.add_argument(
        '-kfold_idx',
        type = int,
        default = 0,
        help = "kfold 인덱스")
    
    parser.add_argument(
        '-datetime',
        type = str,
        default = datetime.now().strftime("%Y%m%d_%H%M%S"),
        help = 'generation log를 위한 시간 표시')

    args = parser.parse_args()
    
    return args