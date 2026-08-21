# Image Search CNN

The purpose of this tool is to build a database index of images, then allow an image to be compared to the database to find the closest matches. It also includes a rotation classifier that can detect whether an image is upside down or sideways.

[CLIP](https://github.com/openai/CLIP) is used to extract the image features as a vector.

[FAISS](https://github.com/facebookresearch/faiss) is used to compare image feature vectors to find the closest matches.

## Requirement
This project requires conda as it is the only easy way to install FAISS.

### Installing Miniconda on Windows / Mac

Install Miniconda from https://www.anaconda.com/download

### Installing Miniconda (conda) on a Linux server

On a headless Linux server, download and run the official installer script instead of using the graphical installer:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept the license, keep the default install location, and let the installer initialise conda for your shell when prompted. Then reload your shell so the `conda` command is available:

```
source ~/.bashrc
```

Verify it installed correctly:

```
conda --version
```

## Quick Start

```
git clone https://github.com/Shadercloud/image_search_cnn
cd image_search_cnn
conda env create -f environment.yml
conda activate image_search_cnn2
cp config_sample.yml config.yml
python .
```

Edit `config.yml` and set `images.path` to the directory your images are stored in on disk (this is used to resolve relative image paths returned from search results, e.g. for the `compare` options).

## Command Line Arguments

| Argument      | Default    | Description                                                              |
|---------------|------------|---------------------------------------------------------------------------|
| --extractor   | clip       | Feature extractor to use (`clip` or `resnet`)                            |
| --host        | localhost  | Webserver host (overrides `webserver.host` in config.yml)                |
| --port        | 8080       | Webserver port (overrides `webserver.port` in config.yml)                |
| --verbose, -v | 0          | Level of log output (0 = not much, 1 = info, 2 = debug)                  |
| --output      | 100        | Print a progress count every N images processed                          |
| --update      | (off)      | Update an image's features if it already exists, instead of skipping it |
| --threads     | 2          | Number of threads to use when processing a directory of images          |

## Database
All image vectors are store in an SQLite database file in the data directory.

FAISS index files are also stored in the data directory

## Webserver

This program runs basic web server on http://localhost:8080

In my case I did not want to be importing Tensorflow / torch and loading the database with every image search.  So instead setup a webserve that runs all the time.  It is expected that your program (written to whichever language) interacts with this program locally via http.

This webserver is not designed to be end-user facing and is not secured for that purpose.

The following endpoints are available over http request:

### Add Images

You can either specify an individual image or a directory.

```
http://localhost:8080/add?image=\Path\to\your\image.jpg
```

*Note that this may take a very long time to run depending on the number of images, you can see the progress from the output on the console.*

### Search Image

Once you have a database built you can search it using:

```
http://localhost:8080/search?image=\Path\to\your\image.jpg&limit=10&compare=basic&compare=sift
```

**GET Parameters:**

    limit=10        | This allows you to limit the number of results returned
    compare=basic   | These are optional comparators which can to used to enhance the results  

**Returns Example:** 
```json
{
    "results": [
        {
            "image": "/Path/To/Image/376093-22123062Fr.jpg",
            "distance": 0.600092172622681
        },
        {
            "image": "/Path/To/Image/376093-22123063Fr.jpg",
            "distance": 0.808739006519318
        }
    ]
}
```

### Remove Image

Removes an image from the database by filename (matched against the basename used when it was added).

```
http://localhost:8080/remove?image=image.jpg
```

### Detect Rotation

Predicts whether an image is rotated, using a MobileNetV2 classifier (`models/rotation_classifier.pth`) trained to recognise 0, 90, 180 and 270 degree rotations. This does not require the image to already be in the database.

```
http://localhost:8080/rotation?image=\Path\to\your\image.jpg
```

**Returns Example:**
```json
{
    "rotation": "90"
}
```

### Get Stats

Get information about how many images are in the database.

```
http://localhost:8080/stats
```

# GPU

If you have a GPU with Cuda but code is still running on the CPU make sure you install the cuda version of torch:

```
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

You can run the `test_cuda.py` script to see if Cuda is being correctly detected:

```
python test_cuda.py
```

*Note that it is significantly faster to run this on a GPU (ie 10-50 times faster)*

# Comparators

I created some comparator functions which take the CNN returns and then compare each image to the search image.  I have found the `sift` comparator to be especially good.

| Comparator   | Returns                                                |
|--------------|--------------------------------------------------------|
| basic        | 1 = identical, -1 = completely different               |
| sift         | Returns 0 to big number (big number is best)           |
| ssim         | 1 = identical, -1 = completely different               |
| histogram    | 1 = identical, 0 = no match                            |
| orb          | Returns 0 to 1, Lower is better (0 = identical images) |
 
