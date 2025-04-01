# Image Search CNN

The purpose of this tool is to build a database index of images, then allow an image to be compared to the database to find the closest matches.

[CLIP](https://github.com/openai/CLIP) is used to extract the image features as a vector.

[FAISS](https://github.com/facebookresearch/faiss) is used to compare image feature vectors to find the closest matches.

## Requirement
This project requires conda as it is the only easy way to install FAISS.

Install miniconda from https://www.anaconda.com/download

## Quick Start

```
git clone https://github.com/Shadercloud/image_search_cnn
cd image_search_cnn
conda env create -f environment.yml
conda activate
python ./
```

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
 
