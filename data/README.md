# Data Format

Datasets should be placed in this directory. The directory structure should be as follows:

```bash
dataset_name (dir)
    ├── train.jsonl
    ├── test.jsonl
    ├── val.jsonl
    ├── images (dir)
    ├── config.json
```

The jsonl files should contain a list of dictionaries, where each dictionary represents a single example. The keys of the dictionary should be the following:

- `id`: A unique identifier for the example.
- `text`: The text of the example.
- `label`: The numeric label of the example.

The images directory should contain the images for the examples. The images should be named with the id of the example they correspond to. For example, if the id of an example is `123`, then the image should be named `123.jpg`.

The config.json file should contain a dictionary with the following keys:

- `num_classes`: The number of classes in the dataset.
- `labels`: A dict mapping the numeric labels to their string representations.