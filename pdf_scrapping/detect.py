import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt
import os
import numexpr as ne

os.environ['NUMEXPR_MAX_THREADS'] = '10'
ne.set_num_threads(4)

# Instantiate the deepdoctection analyzer
analyzer = dd.get_dd_analyzer()

# Analyze the PDF document
df = analyzer.analyze(path="ktu.pdf")

# Reset the state (trigger some initialization)
df.reset_state()

# Specify the directory to save the images
output_directory = "output_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through each page in the document
for page_number, page in enumerate(df):
    # Get the visualization of the page
    image = page.viz()

    # Save the image with a unique name based on the page number
    image_filename = os.path.join(output_directory, f"page_{page_number + 1}.png")
    plt.imsave(image_filename, image)

    # Plot the visualization (optional)
    plt.figure(figsize=(300, 200))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    # Print the text content of the page (optional)
    print(page.text)

print("Images saved in:", output_directory)
