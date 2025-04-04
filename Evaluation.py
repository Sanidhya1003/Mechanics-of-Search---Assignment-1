import subprocess

# Full path to the JAR file
jar_path = r"D:\Practicum\Mechanics of Search\jtreceval\jtreceval-master\target\jtreceval-0.0.5-jar-with-dependencies.jar"

# Paths to input files
qrel_file = r"D:\Practicum\Mechanics of Search\cranqrel.trec.txt"
results_file = r"D:\Practicum\Mechanics of Search\unigram_results.txt"

# Define the command
command = [
    "java", "-jar", jar_path,
    "-m", "P.5", "-m", "map", "-m", "ndcg",
    qrel_file, results_file
]

# Run the command
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)  # Print standard output
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)  # Print any errors if the command fails
