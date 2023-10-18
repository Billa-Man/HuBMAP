from PIL import Image

def tiff_to_png(file_name):
    
    tiff_image_path = "./hubmap-hacking-the-human-vasculature/train/" + str(file_name) + ".tif"
    tiff_image = Image.open(tiff_image_path)
    destination_path = "./kaggle/working/temp_images/" + file_name + ".png"
    tiff_image.save(destination_path, 'PNG')

def vertices_to_txt(file_id, annotations, list_of_vertices):
    
    file_contents = []

    for i in range(len(annotations)):

        yolo_format = []
        flag = 1

        if annotations[i]['type'] == 'glomerulus':
            yolo_format.append(str(1))
            flag = 1
        elif annotations[i]['type'] == 'blood_vessel':
            yolo_format.append(str(0))
            flag = 1
        else:
            flag = 0


        if (flag):

            list_of_vertices = annotations[i]['coordinates'][0]
            for vertex in list_of_vertices:
                yolo_format.append(str(vertex[0]/512))
                yolo_format.append(str(vertex[1]/512))

        yolo_format = " ".join(yolo_format)

        file_contents.append(yolo_format)

    file_name = "./kaggle/working/temp_labels/" + str(file_id) + ".txt"

    with open(file_name, "w") as file:
        if (len(file_contents) == 0):
            pass
        elif (len(file_contents) == 1):
            file.write(str(file_contents[-1]))
        else:
            for k in range(len(file_contents)-1):
                file.write(str(file_contents[k]) + "\n")

            file.write(str(file_contents[-1]))
            
    return 0