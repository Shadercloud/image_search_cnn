# Image Search CNN

The purpose of this tool is to build a database index of images, then allow an image to be compared to the database to find the closest matches.

## Quick Start

```
git clone https://github.com/Shadercloud/image_search_ccn
cd image_search
pip install -r /path/to/requirements.txt

python ./
```

## Database

The data is store in `pickle` files in the `/data` directory.  This will be automatically loaded when the program is started.  You do not need to re-add images each times.

Note that each extractor has its own database file as data from one extractor is not comparable with data from another extractor. 

## Extractor

The program is set up with multiple extractor classes, currently a "clip" or "resnet" version.  For my use case I found the [CLIP](https://github.com/openai/CLIP) model to work much better (which is the default extractor).

Start the server with `python ./ --extractor resnet` if you want to use the other model.

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
 
