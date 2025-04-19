import json
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont

with open("./vendor_similarity_scores.json", "r") as f:
    data = json.load(f)

def table_txt():
    table_data = []
    headers = ["Sector Group", "SS (Mean ± Std)", "Jaccard (Var)", "ROUGE-1 (Var)", "ROUGE-2 (Var)", "ROUGE-L (Var)", "BLEU (Var)"]

    for group in data["groups"]:
        sectors = ", ".join(group["sectors"])  # Convert sector list to string
        scores = group["similarity_scores"]
        row = [
            sectors,
            f"{scores['semantic_similarity_mean']:.4f} ± {scores['semantic_similarity_std']:.4f}",
            f"{scores['jaccard_similarity']:.4f} ({scores['jaccard_variance']:.4f})",
            f"{scores['rouge1_f1']:.4f} ({scores['rouge1_variance']:.4f})",
            f"{scores['rouge2_f1']:.4f} ({scores['rouge2_variance']:.4f})",
            f"{scores['rougeL_f1']:.4f} ({scores['rougeL_variance']:.4f})",
            f"{scores['bleu_score']:.4f} ({scores['bleu_variance']:.4f})"
        ]
        table_data.append(row)

    # Generate table using tabulate
    table = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")

    # Print table to console
    print(table)

    # Save table to a text file
    with open("vendor_similarity_table.txt", "w") as f:
        f.write(table)

def table_image():
    table_data = []
    headers = ["Sector Group", "Semantic Similarity", "Jaccard ", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    for group in data["groups"]:
        sectors = ", ".join(group["sectors"])  # Convert sector list to string
        scores = group["similarity_scores"]
        row = [
            sectors,
            f"{scores['semantic_similarity_mean']:.4f} ± {scores['semantic_similarity_std']:.4f}",
            f"{scores['jaccard_similarity']:.4f} ({scores['jaccard_variance']:.4f})",
            f"{scores['rouge1_f1']:.4f} ({scores['rouge1_variance']:.4f})",
            f"{scores['rouge2_f1']:.4f} ({scores['rouge2_variance']:.4f})",
            f"{scores['rougeL_f1']:.4f} ({scores['rougeL_variance']:.4f})",
            f"{scores['bleu_score']:.4f} ({scores['bleu_variance']:.4f})"
        ]
        table_data.append(row)

    # Image settings
    font_size = 16
    cell_padding = 10
    line_spacing = 5
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  # Adjust path if needed
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate column widths and row heights
    col_widths = []
    for col_idx in range(len(headers)):
        col_values = [row[col_idx] for row in [headers] + table_data]
        max_width = max(font.getbbox(text)[2] for text in col_values) + 2 * cell_padding
        col_widths.append(max_width)

    row_height = font.getbbox("A")[3] + 2 * cell_padding + line_spacing
    total_width = sum(col_widths)
    total_height = row_height * (len(table_data) + 1)  # +1 for header

    # Create image
    image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw table
    # Header
    for col_idx, header in enumerate(headers):
        x = sum(col_widths[:col_idx]) + cell_padding
        y = cell_padding
        draw.text((x, y), header, font=font, fill="black")
    draw.line([(0, row_height), (total_width, row_height)], fill="black", width=2)

    # Data rows
    for row_idx, row in enumerate(table_data):
        y = (row_idx + 1) * row_height
        for col_idx, cell in enumerate(row):
            x = sum(col_widths[:col_idx]) + cell_padding
            draw.text((x, y + cell_padding), cell, font=font, fill="black")
        draw.line([(0, y), (total_width, y)], fill="black", width=1)

    # Vertical lines
    x = 0
    for width in col_widths:
        draw.line([(x, 0), (x, total_height)], fill="black", width=1)
        x += width
    draw.line([(total_width, 0), (total_width, total_height)], fill="black", width=1)

    # Save image
    image.save("vendor_similarity_table.png")

table_txt()
table_image()