from PIL import Image, ImageFilter, ImageDraw
import os

path_to_data = "/data/images/error_test/01/"
path_to_output = "/data/images/error_test/block01/"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''
p = Augmentor.Pipeline(path_to_data)

# pipline
p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.4)
p.rotate90(probability=0.1)
num_of_sample = int(1e2)
p.sample(num_of_sample)
'''

# TODO: Blur
def blur(img_path):
    im = Image.open(img_path)
    im = im.filter(ImageFilter.BLUR)
    return im



# TODO: Contour
def contour(img_path):
    im = Image.open(img_path)
    im = im.filter(ImageFilter.CONTOUR)
    return im



# TODO: Mosaic
def mosaic(img_path):
    im = Image.open(img_path)
    width, height = im.size
    granularity = 3
    new_img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    for x in range(0, width, granularity):
        for y in range(0, height, granularity):
            r, g, b = im.getpixel((x, y))
            draw.rectangle([(x, y), (x + granularity, y+granularity)], fill=(r, g, b), outline=None)

    return new_img



# TODO: Block
def block(img_path):
    im = Image.open(img_path)
    width, height = im.size
    topLeft = (0.45, 0.45)        # Persentage of weight and height
    bottomRight = (0.55, 0.55)
    draw = ImageDraw.Draw(im)
    draw.rectangle([(int(width*topLeft[0]), int(height*topLeft[1])), (int(width*bottomRight[0]), int(height*bottomRight[1]))], fill=(0, 0, 0), outline=None)
    return im



def RGB(img_path):
    return Image.open(img_path).convert("RGB")


# TODO: Augment the image in a folder
def augment_folder(path_to_data, path_to_output, func):
    file_list = os.listdir(path_to_data)
    for file in file_list:
        if os.path.exists(path_to_output + file):
            continue
        try:
            new_img = func(path_to_data + file)
            new_img.save(path_to_output + file)
        except:
            continue



def augment_vggdata():
    mosaic_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/mosaic_images/'
    block_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/block_images/'
    blur_path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/blur_images/'
    path = '/home/SENSETIME/dengyang/PycharmProjects/vgg-face.pytorch/images/vgg_face_dataset/new_images/'
    folder_list = os.listdir(path)
    for i, folder in enumerate(folder_list):
        print('Augmenting folder: %d/%d' %(i+1, len(folder_list)), flush=1)
        folder_path = path + folder + '/'
        mosaic_folder_path = mosaic_path + folder + '/'
        block_folder_path = block_path + folder + '/'
        blur_folder_path = blur_path + folder + '/'


        mkdir(mosaic_folder_path)
        mkdir(block_folder_path)
        mkdir(blur_folder_path)

        augment_folder(folder_path, folder_path, func=RGB)

        augment_folder(folder_path, mosaic_folder_path, func=mosaic)
        augment_folder(folder_path, block_folder_path, func=block)
        augment_folder(folder_path, blur_folder_path, func=blur)




if __name__ == '__main__':
    # if not os.path.exists(path_to_output):
    #     os.makedirs(path_to_output)
    # Choose the func {mosaic, block, blur, contour}
    #augment_folder(path_to_data, path_to_output, func=block)
    augment_vggdata()