import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import struct
import cv2 as cv
import constants
from utils import iou_width_height
# General global variables for reading data input
base_directory_for_packs = '/home/ryan/TrainingData/new_object_detection'
inner_pack_iterator_directory = '/depth_images'
inner_pack_double_array_path = "/double_array/data.bin"
input_image_file_extension = '.TIFF'
# Global variables for Reading .bin File
number_of_possible_bounding_boxes = 10
variables_for_each_bounding_box_from_input = 11
double_byte_size = 8
byte_size_of_double_array = number_of_possible_bounding_boxes * variables_for_each_bounding_box_from_input * double_byte_size
byte_size_bounding_box = variables_for_each_bounding_box_from_input * double_byte_size

class CustomImageDataset(Dataset):
    def __init__(self):
        self.total_images = 0
        self.item_pack_map = {}
        self.item_inner_pack_map = {}
        self.current_file_loaded = None
        self.current_file_name = ""
        self.split_sizes = constants.split_sizes
        self.anchors = []
        for split_size in range(len(constants.anchor_boxes)):
            self.anchors = self.anchors + constants.anchor_boxes[split_size]
        self.anchors = torch.tensor(self.anchors)
        self.number_of_anchors = self.anchors.shape[0]
        self.number_of_anchors_per_scale = self.number_of_anchors // len(constants.anchor_boxes)
        self.ignore_iou_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # Set File Count and Maps To Files
        subdirectories = next(os.walk(base_directory_for_packs))[1]
        for directory in subdirectories:
            # Assumes there is an image data type in dataloader
            file_count = len(next(os.walk(base_directory_for_packs + "/" + directory + inner_pack_iterator_directory))[2])
            for file in range(1,file_count+1):
                self.item_pack_map[self.total_images] = int(directory)
                self.item_inner_pack_map[self.total_images] = file
                self.total_images += 1

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image = torch.zeros(1, constants.model_image_width, constants.model_image_height, dtype=torch.float)
        # 6 = [ Prob, x, y, w, h, class]
        label = [torch.zeros((self.number_of_anchors_per_scale, scale_size, scale_size, 6)) for scale_size in self.split_sizes]

        # Get Input Image
        current_image_file_path =base_directory_for_packs + "/" + str(self.item_pack_map[idx]) + inner_pack_iterator_directory + "/" + str(self.item_inner_pack_map[idx]) + input_image_file_extension
        unmodified_camera_image = cv.imread(current_image_file_path, cv.IMREAD_ANYDEPTH)
        resized_image = cv.resize(unmodified_camera_image,(constants.model_image_width,constants.model_image_height))
        camera_tensor = torch.from_numpy(resized_image)
        removed_nan_tensor = torch.nan_to_num(camera_tensor.to(self.device), nan=100.0, posinf=100.0, neginf=100.0).unsqueeze(0)
        # normalized_input = removed_nan_tensor.mul(-1/5)
        # normalized_input = torch.sigmoid(removed_nan_tensor.mul(-1/5))
        image = removed_nan_tensor

        # Get Label
        for bounding_box_index in range(number_of_possible_bounding_boxes):

            name_of_file_to_read = base_directory_for_packs + "/" + str(self.item_pack_map[idx]) + inner_pack_double_array_path
            if self.current_file_name != name_of_file_to_read:
                self.current_file_name = name_of_file_to_read
                self.current_file_loaded = Path(self.current_file_name).read_bytes()
            start_idx_in_bin_file = self.item_inner_pack_map[idx]-1

            classification_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 0 * double_byte_size
            min_x_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 1 * double_byte_size
            max_x_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 2 * double_byte_size
            min_y_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 3 * double_byte_size
            max_y_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 4 * double_byte_size
            # x_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 5 * double_byte_size
            # y_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 6 * double_byte_size
            # z_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 7 * double_byte_size
            # roll_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 8 * double_byte_size
            # pitch_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 9 * double_byte_size
            # yaw_byte_start_index = byte_size_of_double_array * start_idx_in_bin_file + byte_size_bounding_box * bounding_box_index + 10 * double_byte_size

            classification = struct.unpack('d', self.current_file_loaded[classification_byte_start_index:classification_byte_start_index + double_byte_size])[0]
            min_x = struct.unpack('d', self.current_file_loaded[min_x_byte_start_index:min_x_byte_start_index + double_byte_size])[0]
            max_x = struct.unpack('d', self.current_file_loaded[max_x_byte_start_index:max_x_byte_start_index + double_byte_size])[0]
            min_y = struct.unpack('d', self.current_file_loaded[min_y_byte_start_index:min_y_byte_start_index + double_byte_size])[0]
            max_y = struct.unpack('d', self.current_file_loaded[max_y_byte_start_index:max_y_byte_start_index + double_byte_size])[0]
            # x = struct.unpack('d', self.current_file_loaded[x_byte_start_index:x_byte_start_index + double_byte_size])[0]
            # y = struct.unpack('d', self.current_file_loaded[y_byte_start_index:y_byte_start_index + double_byte_size])[0]
            # z = struct.unpack('d', self.current_file_loaded[z_byte_start_index:z_byte_start_index + double_byte_size])[0]
            # roll = struct.unpack('d', self.current_file_loaded[roll_byte_start_index:roll_byte_start_index + double_byte_size])[0]
            # pitch = struct.unpack('d', self.current_file_loaded[pitch_byte_start_index:pitch_byte_start_index + double_byte_size])[0]
            # yaw = struct.unpack('d', self.current_file_loaded[yaw_byte_start_index:yaw_byte_start_index + double_byte_size])[0]

            if classification >= 0:
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                height_of_box = abs(max_y - min_y)
                width_of_box = abs(max_x - min_x)
                iou_anchors = iou_width_height(torch.tensor([width_of_box, height_of_box]), self.anchors)
                anchor_indicies = iou_anchors.argsort(descending=True, dim=0)

                has_anchor = [False] * 3
                for anchor_idx in anchor_indicies:
                    anchor_idx = anchor_idx.item()
                    scale_idx = anchor_idx // self.number_of_anchors_per_scale
                    anchor_idx_on_scale = anchor_idx % self.number_of_anchors_per_scale
                    scale = self.split_sizes[scale_idx]
                    cell_row = int(center_y * scale)
                    cell_col = int(center_x * scale)
                    anchor_taken = label[scale_idx][anchor_idx_on_scale][cell_row][cell_col][0]
                    if not anchor_taken and not has_anchor[scale_idx]:
                        x_cell = center_x * scale - cell_col
                        y_cell = center_y * scale - cell_row
                        width_cell = width_of_box * scale
                        height_cell = height_of_box * scale
                        label[scale_idx][anchor_idx_on_scale][cell_row][cell_col] = torch.tensor([1,x_cell,y_cell,width_cell,height_cell,classification])
                        has_anchor[scale_idx] = True
                    elif not anchor_taken and iou_anchors[anchor_idx]>self.ignore_iou_threshold:
                        label[scale_idx][anchor_idx_on_scale,cell_row,cell_col,0] = -1
        return image, label
