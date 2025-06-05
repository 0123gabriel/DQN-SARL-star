import os

test_cases = [21, 4, 83] # no, dynamic, static
model_dirs = ['trained_models/train_11_4000ed_sarl', 'trained_models/train_12_2000ed_sarl']
policy = 'sarl'

for model_dir in model_dirs:
    for test_case in test_cases:
        cmd = (
            f"python test.py --policy {policy} "
            f"--model_dir {model_dir} --phase test "
            f"--visualize --test_case {test_case}"
        )
        print(f"Running: {cmd}")
        os.system(cmd)
