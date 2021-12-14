import os
import time
import argparse
import requests
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from requests import exceptions

ap = argparse.ArgumentParser()
ap.add_argument(
    "-q", 
    "--query", 
    required=True,
    help="search query to search Bing Image API for"
)
ap.add_argument(
    "-n", 
    "--max_number_images", 
    required=True,
    help="total number of images to be saved"
)
ap.add_argument(
    "-o", 
    "--output_dir", 
    required=False,
    help="output directory to which images will be saved"
)

# TODO: place a .env file in the same folder with BING_IMG_SEARCH_API_KEY=...
load_dotenv()
API_KEY = os.getenv("BING_IMG_SEARCH_API_KEY")
print(API_KEY)

# number of images to be downloaded in a batch
GROUP_SIZE = 50

URL = "https://api.bing.microsoft.com/v7.0/images/search"

EXCEPTIONS = {
    IOError, 
    FileNotFoundError, 
    exceptions.RequestException, 
    exceptions.HTTPError, 
    exceptions.ConnectionError,
    exceptions.Timeout
}

def crawl_images(term, max_resuls, output_dir, api_key=API_KEY) -> bool:
    if not API_KEY:
        print("Please set BING_IMG_SEARCH_API_KEY")
        return False

    output_dir = Path(output_dir)

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    # initialize the total number of images downloaded thus far
    # change this in case wanting to start from non-zero index
    count = 0
    group_size = min(GROUP_SIZE, max_resuls)
    params = {"q": term, "offset": count, "count": group_size}

    # make the search
    print("[INFO] searching Bing API for '{}'".format(term))
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()

    results = search.json()
    estNumResults = min(results["totalEstimatedMatches"], max_resuls)
    print("[INFO] {} total results for '{}'".format(estNumResults, term))

    # loop over the estimated number of results in `group_size` groups
    for offset in range(count, estNumResults, group_size):
        # update the search parameters using the current offset, then
        # make the request to fetch the results
        print("[INFO] making request for group {}-{} of {}...".format(
            offset, offset + group_size, estNumResults))
        params["offset"] = offset
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print("[INFO] saving images for group {}-{} of {}...".format(
            offset, offset + group_size, estNumResults))

        # loop over the results
        for v in results["value"]:
            # try to download the image
            try:
                # make a request to download the image
                print("[INFO] fetching: {}".format(v["contentUrl"]))
                r = requests.get(v["contentUrl"], timeout=30)

                # build the path to the output image
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]

                dataset_path = output_dir / "raw_data" / term
                path_out_img = dataset_path / f"{str(count).zfill(8)}{ext}"
                dataset_path.mkdir(parents=True, exist_ok=True)

                # write the image to disk
                f = open(path_out_img, "wb")
                f.write(r.content)
                f.close()

            # catch any errors that would not unable us to download the
            # image
            except Exception as e:
                # check to see if our exception is in our list of
                # exceptions to check for
                print(e)
                if type(e) in EXCEPTIONS:
                    print("[INFO] skipping: {}".format(v["contentUrl"]))
                    continue

            try:
                im = Image.open(path_out_img)
            except FileNotFoundError:
                print("[INFO] file not found: {}".format(path_out_img))
                continue
            except OSError:
                print("[INFO] skipping due to OS error: {}".format(path_out_img))
                continue
            except IOError:
                # filename not an image file, so it should be ignored
                print("[INFO] deleting: {}".format(path_out_img))
                os.remove(path_out_img)
                continue

            # update the counter
            count += 1

    return True

if __name__ == '__main__':

    args = vars(ap.parse_args())
    # query string to be searched
    term = args["query"]
    # total number of images to be downloaded
    max_resuls = args["max_number_images"]
    # output directory
    output_dir = Path(args["output_dir"]) if args["output_dir"] else Path.cwd()

    if term and max_resuls:
        startTime = time.time()

        if crawl_images(term, max_resuls, output_dir):
            print('Crawling completed successfully')
        else:
            print('Crawling failed')
    
        print('Total time elapsed: ', time.time() - startTime)
    else:
        print('query string and max number to download not specified, try run the script with "--help" to understand more')