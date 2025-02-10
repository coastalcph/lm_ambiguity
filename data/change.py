import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Keep only rows with '1' status in ambig_status column of a TSV file.")
parser.add_argument("tsv_path", type=str, help="Path to the input TSV file.")
parser.add_argument("--output", type=str, default="true_status_{filename}", help="Path to save the cleaned TSV file (default: true_status_{filename}).")

args = parser.parse_args()

# Load the TSV file
df = pd.read_csv(args.tsv_path, sep="\t", quoting=3, dtype=str, na_filter=False)

# Filter the rows where ambig_status is '1'
df_filtered = df[df["ambig_status"] == "1"]

# Generate output filename
output_filename = args.output.format(filename=args.tsv_path.split('/')[-1])

# Save the modified file
df_filtered.to_csv(output_filename, sep="\t", index=False, quoting=3)

print(f"Filtered file saved as: {output_filename}")

