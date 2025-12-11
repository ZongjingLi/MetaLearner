import re
import csv

def parse_log_to_csv(log_lines, output_csv_path="experiment_accuracy.csv"):
    """
    Parse log lines with acc/test_acc into a CSV file (epoch-wise).
    
    Args:
        log_lines: List of log strings (each line is one log entry)
        output_csv_path: Path to save the CSV file
    """
    # Regex pattern to extract acc and test_acc values from log lines
    pattern = r"acc:([0-9.]+), test_acc: ([0-9.]+)"
    
    # Extract data and assign epoch numbers (1-based)
    data = []
    for epoch, line in enumerate(log_lines, start=1):
        match = re.search(pattern, line)
        if match:
            train_acc = float(match.group(1))
            test_acc = float(match.group(2))
            data.append({
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc
            })
    
    # Check if data was extracted
    if not data:
        print("❌ No acc/test_acc values found in log lines.")
        return
    
    # Write to CSV
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_acc", "test_acc"])
        writer.writeheader()  # Write column headers
        writer.writerows(data)
    
    print(f"✅ Successfully saved {len(data)} epochs to {output_csv_path}")
    # Print preview
    print("\n📊 CSV Preview (first 5 rows):")
    for row in data[:5]:
        print(f"Epoch {row['epoch']}: Train Acc = {row['train_acc']:.4f}, Test Acc = {row['test_acc']:.4f}")

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Paste your log lines here (replace with your actual log)
    log_lines = """
[08 19:37:00 483@model.py:SearchExecutor] acc:0.6987307071685791, test_acc: 0.9062497019767761
[08 19:37:04 483@model.py:SearchExecutor] acc:0.8613282442092896, test_acc: 0.7499991655349731
[08 19:37:08 483@model.py:SearchExecutor] acc:0.97509765625, test_acc: 1.0
[08 19:37:13 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:17 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:21 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:25 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:29 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:33 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:37 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:41 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:46 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:49 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:53 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:37:57 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:38:01 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:38:05 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:38:09 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:38:13 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
[08 19:38:17 483@model.py:SearchExecutor] acc:1.0, test_acc: 1.0
""".split("\n")
    
    # Parse log and save to CSV
    parse_log_to_csv(log_lines, output_csv_path="outputs/logs/expr_1024/expr_1024_9.csv")