import matplotlib.pyplot as plt

# Assuming you have a list of entities and their respective scores for degree and modularity
entities = ['M-Coarse', 'Ace-event', 'Ace05', 'M-Granular', 'Genia', 'SciErc']
# Replace with actual degree scores
degree_scores = [2.47, 1.23, 1.29, 1.76, 0.99, 2.18]
# Replace with actual modularity scores
modularity_scores = [0.04, 0.16, 0.18, 0.01, 0.07, 0.33]

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# Increase the size of the dots
plt.scatter(degree_scores, modularity_scores, s=300, c='orange')

# Add labels for each point (optional)
for i, txt in enumerate(entities):
    # Increase the size of the labels
    plt.annotate(
        txt, (degree_scores[i], modularity_scores[i]), fontsize=16)

# Add title and labels for the axes
plt.title('Degree vs Modularity for Entities', fontsize=16)
plt.xlabel('Degree Score', fontsize=14)
plt.ylabel('Modularity Score', fontsize=14)

# Show the plot
plt.show()
