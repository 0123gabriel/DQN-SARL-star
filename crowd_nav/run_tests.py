import os

test_cases = [21, 4, 83]
model_dirs = ['trained_models/train_1_4000ed_cadrl', 'trained_models/train_2_2000ed_cadrl']
policy = 'cadrl'

for model_dir in model_dirs:
    for test_case in test_cases:
        cmd = (
            f"python test.py --policy {policy} "
            f"--model_dir {model_dir} --phase test "
            f"--visualize --test_case {test_case}"
        )
        print(f"Running: {cmd}")
        os.system(cmd)
