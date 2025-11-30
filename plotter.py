import matplotlib.pyplot as plt

# Raw data: each line is [x1, y1, x2, y2, mark]
input = """
35 -9 28 -5 vertical wall
35 -9 29 -5 vertical wall
35 -9 30 -5 vertical wall
35 -9 31 -5 vertical wall
35 -9 32 -5 vertical wall
35 -9 33 -5 vertical wall
35 -9 34 -5 vertical wall
35 -9 35 -5 vertical wall
35 -9 36 -5 vertical wall
35 -9 28 -6 horizontal wall
35 -9 29 -6 staircase up
35 -9 30 -6 floor
35 -9 31 -6 boulder
35 -9 32 -6 floor
35 -9 33 -6 floor
35 -9 34 -6 floor
35 -9 35 -6 floor
35 -9 36 -6 horizontal wall
35 -9 28 -7 horizontal wall
35 -9 29 -7 boulder
35 -9 30 -7 floor
35 -9 31 -7 boulder
35 -9 32 -7 sink
35 -9 33 -7 floor
35 -9 34 -7 floor
35 -9 35 -7 floor
35 -9 36 -7 horizontal wall
35 -9 28 -8 horizontal wall
35 -9 29 -8 floor
35 -9 30 -8 floor
35 -9 31 -8 weapon
35 -9 32 -8 boulder
35 -9 33 -8 floor
35 -9 34 -8 floor
35 -9 35 -8 fountain
35 -9 36 -8 horizontal wall
35 -9 28 -9 vertical wall
35 -9 29 -9 vertical wall
35 -9 30 -9 vertical wall
35 -9 31 -9 floor
35 -9 32 -9 weapon
35 -9 33 -9 floor
35 -9 34 -9 floor
35 -9 36 -9 horizontal wall
35 -9 30 -10 horizontal wall
35 -9 31 -10 floor
35 -9 32 -10 floor
35 -9 33 -10 floor
35 -9 34 -10 wand
35 -9 35 -10 staircase down
35 -9 36 -10 floor
35 -9 37 -10 floor
35 -9 38 -10 floor
35 -9 39 -10 floor
35 -9 40 -10 floor
35 -9 30 -11 vertical wall
35 -9 31 -11 vertical wall
35 -9 32 -11 vertical wall
35 -9 33 -11 vertical wall
35 -9 34 -11 vertical wall
35 -9 35 -11 vertical wall
35 -9 36 -11 vertical wall
35 -9 37 -11 vertical wall
35 -9 38 -11 vertical wall
35 -9 39 -11 vertical wall
"""

def parse_input(input_text):
    # Split input into lines, filter out empty lines
    lines = [line.strip() for line in input_text.split('\n') if line.strip()]
    data_array = []
    
    for line in lines:
        # Split each line into parts (handles multiple spaces)
        parts = line.split()
        # Extract numeric values (convert to int) and combine multi-word marks
        x1 = int(parts[0])
        y1 = int(parts[1])
        x2 = int(parts[2])
        y2 = int(parts[3])
        mark = ' '.join(parts[4:])  # Handles marks with spaces (e.g., "staircase down")
        # Append to array
        data_array.append([x1, y1, x2, y2, mark])
    
    return data_array

# Run parsing and get the structured array
data = parse_input(input)



# Extract coordinates and marks
x1 = [row[0] for row in data]
y1 = [row[1] for row in data]
x2 = [row[2] for row in data]
y2 = [row[3] for row in data]
marks = [row[4] for row in data]

# Create plot
plt.figure(figsize=(12, 8))

# Plot (x1, y1) reference point (all (29,7))
plt.scatter(x1, y1, color='red', s=200, marker='*', label='Reference (x1, y1) = (29, 7)', zorder=5)

# Plot (x2, y2) nodes with unique colors for each mark type
mark_types = list(set(marks))
colors = plt.cm.tab10(range(len(mark_types)))  # Assign unique colors
mark_color_map = dict(zip(mark_types, colors))

# Plot each node and add label
for x, y, mark in zip(x2, y2, marks):
    color = mark_color_map[mark]
    plt.scatter(x, y, color=color, s=100, alpha=0.8, zorder=3)
    # Add mark label (offset slightly to avoid overlap)
    plt.annotate(mark, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Customize plot
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.title('Scatter Plot with Node Marks (x2, y2) and Reference (x1, y1)', fontsize=14, pad=15)

# Create legend for mark types
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=mark) 
                   for mark, color in mark_color_map.items()]
# Add reference point to legend
legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Reference (29, 7)'))

plt.legend(handles=legend_elements, fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  # Adjust layout to fit labels

# Show plot
plt.show()