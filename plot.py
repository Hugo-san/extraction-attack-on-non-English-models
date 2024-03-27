import pandas as pd
import matplotlib.pyplot as plt

# List of your datasets
data_types = ['S_XL', 'zlib']  # Replace with your actual data type names

for data_type in data_types:
    csv_file = f'{data_type}_best_samples_scores.csv'

    # Load the data from CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Filter out the samples that have been memorized (Occurence > 0)
    memorized_data = data[data['Occurence'] > 0]

    # Filter out the samples that have been selected but not memorized (Occurence == 0)
    selected_data = data[data['Occurence'] == 0]

    # Plotting selected but not memorized samples
    plt.scatter(selected_data['PPL-XL'], selected_data['Zlib'], alpha=0.7, label='Selected', color='red')

    # Plotting memorized samples
    plt.scatter(memorized_data['PPL-XL'], memorized_data['Zlib'], alpha=0.7, label='Memorized', color='blue')

    # Adding labels and title
    plt.xlabel('GPT-2 Perplexity')
    plt.ylabel('zlib Entropy')
    plt.title(f'Perplexity vs Entropy - {data_type}')

    # Adding a legend
    plt.legend()

    # Save the plot to a file
    plt.savefig(f'{data_type}_plot.png')

    # Show the plot
    plt.show()

    # Clear the figure for the next dataset
    plt.clf()
