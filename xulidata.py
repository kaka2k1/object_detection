# # # import os

# # # def process_txt_file(txt_file_path):
# # #     valid_lines = []
# # #     with open(txt_file_path, 'r') as file:
# # #         for line in file:
# # #             label = line.strip().split(' ')[0]
# # #             if label in ['0', '1', '2', '3', '4', '5', '6']:
# # #                 valid_lines.append(line)

# # #     if len(valid_lines) == 0:
# # #         os.remove(txt_file_path)
# # #     else:
# # #         with open(txt_file_path, 'w') as file:
# # #             file.writelines(valid_lines)

# # # def process_labels_directory(labels_directory):
# # #     for file_name in os.listdir(labels_directory):
# # #         if file_name.endswith('.txt'):
# # #             txt_file_path = os.path.join(labels_directory, file_name)
# # #             process_txt_file(txt_file_path)

# # # def main():
# # #     data_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\exp5"  # Thay đổi đường dẫn tới thư mục chứa data ở đây
# # #     labels_directory = os.path.join(data_directory, "labels")

# # #     process_labels_directory(labels_directory)

# # # if __name__ == "__main__":
# # #     main()

# # import os

# # # Đường dẫn đến thư mục chứa các file txt
# # folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data2\\labels"

# # # Lấy danh sách tên các file trong thư mục
# # file_list = os.listdir(folder_path)

# # # Duyệt qua từng tên file
# # for file_name in file_list:
# #     # Kiểm tra nếu file có phần mở rộng là .txt
# #     if file_name.endswith(".txt"):
# #         # Chuyển tên file bằng cách thêm 'magnitude_spectrum_' vào trước tên file hiện tại
# #         new_file_name = "magnitude_spectrum_" + file_name

# #         # Đường dẫn đến file gốc và file mới
# #         old_file_path = os.path.join(folder_path, file_name)
# #         new_file_path = os.path.join(folder_path, new_file_name)

# #         # Tiến hành đổi tên file
# #         os.rename(old_file_path, new_file_path)




# # import os
# # import cv2

# # image_path = r"E:\PROJECT_IN_WISDOM\Object_Detect\data\image2\magnitude_spectrum_000000000108.jpg"
# # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # # Kiểm tra xem hình ảnh đã đọc thành công chưa
# # if img is None:
# #     print("Không thể đọc hình ảnh. Kiểm tra đường dẫn tập tin.")
# # else:
# #     # Hiển thị hình ảnh
# #     cv2.imshow("Hình ảnh", img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


# import os
# import numpy as np
# import cv2

# def compute_spectrum(image_path, target_size=(1400, 1000)):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
#     # Apply Fourier Transform
#     f_transform = np.fft.fft2(img)
#     f_transform_shifted = np.fft.fftshift(f_transform)
    
#     # Compute the magnitude spectrum (logarithmic scale for visualization)
#     magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
#     # Normalize and convert magnitude spectrum to uint8 format
#     magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8)
    
#     # Compute the inverse Fourier Transform to get the inverse spectrum
#     f_inverse = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted))
#     img_inverse = np.abs(f_inverse)
    
#     return img, magnitude_spectrum, img_inverse

# # Thư mục chứa các hình ảnh gốc
# images_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\dataset\\images"  # Thay đổi đường dẫn tới thư mục chứa hình ảnh

# # Thư mục chứa các hình ảnh quang phổ
# output_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\dataset2\\images"  # Thay đổi đường dẫn tới thư mục chứa hình ảnh quang phổ

# # Tạo thư mục nếu nó chưa tồn tại
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Xử lý từng hình ảnh trong thư mục "images" và lưu kết quả vào thư mục "image2"
# for image_file in os.listdir(images_directory):
#     image_path = os.path.join(images_directory, image_file)
#     img, magnitude_spectrum, _ = compute_spectrum(image_path)

#     # Lưu hình quang phổ vào thư mục "image2"
#     output_path = os.path.join(output_directory, f"magnitude_spectrum_{image_file}")
#     cv2.imwrite(output_path, magnitude_spectrum)




# # import os
# # import numpy as np
# # import cv2

# # def compute_spectrum(image_path, target_size=(1400, 1000)):
# #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
# #     # Apply Fourier Transform
# #     f_transform = np.fft.fft2(img)
# #     f_transform_shifted = np.fft.fftshift(f_transform)
    
# #     # Compute the magnitude spectrum (logarithmic scale for visualization)
# #     magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
# #     # Compute the inverse Fourier Transform to get the inverse spectrum
# #     f_inverse = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted))
# #     img_inverse = np.abs(f_inverse)
    
# #     return img, magnitude_spectrum, img_inverse

# # # Thư mục chứa các hình ảnh gốc
# # images_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data\\images"  # Thay đổi đường dẫn tới thư mục chứa hình ảnh

# # # Thư mục chứa các hình ảnh quang phổ
# # output_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data\\image2"  # Thay đổi đường dẫn tới thư mục chứa hình ảnh quang phổ

# # # Tạo thư mục nếu nó chưa tồn tại
# # if not os.path.exists(output_directory):
# #     os.makedirs(output_directory)

# # # Xử lý từng hình ảnh trong thư mục "images" và lưu kết quả vào thư mục "image2"
# # for image_file in os.listdir(images_directory):
# #     image_path = os.path.join(images_directory, image_file)
# #     img, magnitude_spectrum, _ = compute_spectrum(image_path)

# #     # Lưu hình quang phổ vào thư mục "image2"
# #     output_path = os.path.join(output_directory, f"magnitude_spectrum_{image_file}")
# #     cv2.imwrite(output_path, magnitude_spectrum)

# #     print(f"Saved magnitude spectrum of {image_file} to {output_path}")

# # import os

# # def filter_labels_file(file_path):
# #     lines_to_keep = []
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             # Kiểm tra nếu dòng bắt đầu với kí tự 0, 1, 2, 3, 4, 5 hoặc 6
# #             if line.strip().startswith(('0', '1', '2', '3', '4', '5', '6')):
# #                 # Nếu dòng bắt đầu với 0 hoặc 2, giữ lại 2500 dòng
# #                 if line.strip().startswith(('0', '2')):
# #                     if len(lines_to_keep) < 2500:
# #                         lines_to_keep.append(line)
# #                 else:
# #                     lines_to_keep.append(line)

# #     # Ghi lại các dòng đã lọc vào tập tin
# #     with open(file_path, 'w') as file:
# #         file.writelines(lines_to_keep)

# # def process_labels_folder(labels_folder_path):
# #     # Duyệt qua từng file trong thư mục labels
# #     for file_name in os.listdir(labels_folder_path):
# #         file_path = os.path.join(labels_folder_path, file_name)
# #         if os.path.isfile(file_path) and file_name.endswith('.txt'):
# #             filter_labels_file(file_path)

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     labels_folder_path = os.path.join(data_folder_path, "labels")

# #     process_labels_folder(labels_folder_path)

# # import os

# # def count_lines_start_with_zero(file_path):
# #     count = 0
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             if line.strip().startswith('2'):
# #                 count += 1
# #     return count

# # def filter_labels(labels_folder_path, max_lines_to_remove=9000):
# #     # Đếm số lượng dòng bắt đầu bằng '0' và sắp xếp các tệp tin theo thứ tự giảm dần
# #     files_info = []
# #     for file_name in os.listdir(labels_folder_path):
# #         file_path = os.path.join(labels_folder_path, file_name)
# #         if os.path.isfile(file_path) and file_name.endswith('.txt'):
# #             lines_with_zero = count_lines_start_with_zero(file_path)
# #             files_info.append((file_path, lines_with_zero))
# #     files_info.sort(key=lambda x: x[1], reverse=True)

# #     # Giữ lại tối đa 77,600 dòng từ tệp tin đầu tiên
# #     max_lines_to_keep = max_lines_to_remove + files_info[0][1]
# #     lines_kept = 0
# #     with open(files_info[0][0], 'r') as file:
# #         lines_to_keep = []
# #         for line in file:
# #             if line.strip().startswith('2') and lines_kept < max_lines_to_keep:
# #                 lines_to_keep.append(line)
# #                 lines_kept += 1

# #     # Ghi lại các dòng đã lọc vào tệp tin đầu tiên
# #     with open(files_info[0][0], 'w') as file:
# #         file.writelines(lines_to_keep)

# #     # Xóa bớt 77,600 dòng từ các tệp tin còn lại
# #     for file_path, lines_with_zero in files_info[1:]:
# #         max_lines_to_keep = max(0, lines_with_zero - max_lines_to_remove)
# #         lines_kept = 0
# #         with open(file_path, 'r') as file:
# #             lines_to_keep = []
# #             for line in file:
# #                 if not line.strip().startswith('2') or lines_kept < max_lines_to_keep:
# #                     lines_to_keep.append(line)
# #                     if line.strip().startswith('2'):
# #                         lines_kept += 1

# #         # Ghi lại các dòng đã lọc vào tệp tin
# #         with open(file_path, 'w') as file:
# #             file.writelines(lines_to_keep)

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     labels_folder_path = os.path.join(data_folder_path, "labels")

# #     filter_labels(labels_folder_path)
# # import os

# # def count_lines_start_with_digit(file_path, digit):
# #     count = 0
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             if line.strip().startswith(digit):
# #                 count += 1
# #     return count

# # def filter_labels(labels_folder_path, max_lines_to_remove=9000):
# #     # Đếm số lượng dòng bắt đầu bằng '2' và sắp xếp các tệp tin theo thứ tự giảm dần
# #     files_info = []
# #     for file_name in os.listdir(labels_folder_path):
# #         file_path = os.path.join(labels_folder_path, file_name)
# #         if os.path.isfile(file_path) and file_name.endswith('.txt'):
# #             lines_with_two = count_lines_start_with_digit(file_path, '2')
# #             files_info.append((file_path, lines_with_two))
# #     files_info.sort(key=lambda x: x[1], reverse=True)

# #     # Giữ lại tối đa 9,000 dòng từ tệp tin đầu tiên
# #     max_lines_to_keep = max_lines_to_remove + files_info[0][1]
# #     lines_kept = 0
# #     with open(files_info[0][0], 'r') as file:
# #         lines_to_keep = []
# #         for line in file:
# #             if line.strip().startswith('2') and lines_kept < max_lines_to_keep:
# #                 lines_to_keep.append(line)
# #                 lines_kept += 1

# #     # Ghi lại các dòng đã lọc vào tệp tin đầu tiên
# #     with open(files_info[0][0], 'w') as file:
# #         file.writelines(lines_to_keep)

# #     # Xóa bớt 9,000 dòng từ các tệp tin còn lại
# #     for file_path, lines_with_two in files_info[1:]:
# #         max_lines_to_keep = max(0, lines_with_two - max_lines_to_remove)
# #         lines_kept = 0
# #         with open(file_path, 'r') as file:
# #             lines_to_keep = []
# #             for line in file:
# #                 if not line.strip().startswith('0') or lines_kept < max_lines_to_keep:
# #                     lines_to_keep.append(line)
# #                     if line.strip().startswith('0'):
# #                         lines_kept += 1

# #         # Ghi lại các dòng đã lọc vào tệp tin
# #         with open(file_path, 'w') as file:
# #             file.writelines(lines_to_keep)

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     labels_folder_path = os.path.join(data_folder_path, "labels")

# #     filter_labels(labels_folder_path)


# # import os

# # def is_file_empty(file_path):
# #     # Kiểm tra xem tệp có dữ liệu hay không
# #     return os.path.exists(file_path) and os.path.getsize(file_path) == 0

# # def remove_empty_files(labels_folder_path):
# #     # Xóa các tệp tin rỗng trong thư mục "labels"
# #     for file_name in os.listdir(labels_folder_path):
# #         file_path = os.path.join(labels_folder_path, file_name)
# #         if os.path.isfile(file_path) and file_name.endswith('.txt') and is_file_empty(file_path):
# #             os.remove(file_path)

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     labels_folder_path = os.path.join(data_folder_path, "labels2")

# #     remove_empty_files(labels_folder_path)
# # import os

# # def keep_files_with_0_and_2(file_path):
# #     has_zero = False
# #     has_two = False
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             if line.strip().startswith('0'):
# #                 has_zero = True
# #             elif line.strip().startswith('2'):
# #                 has_two = True

# #             # Nếu tệp tin đã chứa cả '0' và '2', ta giữ lại tất cả dòng và kết thúc kiểm tra
# #             if has_zero and has_two:
# #                 return

# #     # Nếu tệp tin không chứa cả '0' và '2', ta xóa tệp tin
# #     os.remove(file_path)

# # def process_labels_folder(labels_folder_path):
# #     # Duyệt qua từng file trong thư mục labels
# #     for file_name in os.listdir(labels_folder_path):
# #         file_path = os.path.join(labels_folder_path, file_name)
# #         if os.path.isfile(file_path) and file_name.endswith('.txt'):
# #             keep_files_with_0_and_2(file_path)

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     labels_folder_path = os.path.join(data_folder_path, "labels2")

# #     process_labels_folder(labels_folder_path)

# # import os

# # def find_matching_files(images_folder, labels_folder):
# #     image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
# #     label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

# #     matching_files = []
# #     for image_file in image_files:
# #         image_name = os.path.splitext(image_file)[0]
# #         for label_file in label_files:
# #             label_name = os.path.splitext(label_file)[0]
# #             if image_name == label_name:
# #                 matching_files.append((os.path.join(images_folder, image_file), os.path.join(labels_folder, label_file)))
# #                 break

# #     return matching_files

# # def process_data_folder(data_folder):
# #     images_folder = os.path.join(data_folder, 'images')
# #     labels_folder = os.path.join(data_folder, 'labels')

# #     matching_files = find_matching_files(images_folder, labels_folder)

# #     # Các tệp trùng nhau đã được tìm thấy, có thể thực hiện các thao tác tiếp theo ở đây.
# #     # Ví dụ: in thông tin các cặp tệp trùng nhau
# #     for image_file, label_file in matching_files:
# #         print(f"Matching Pair: Image - {image_file}, Label - {label_file}")

# # if __name__ == "__main__":
# #     data_folder_path = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\data"
# #     process_data_folder(data_folder_path)

# import os
# import cv2

# # Đường dẫn đến thư mục chứa các hình ảnh
# input_folder = r"E:\PROJECT_IN_WISDOM\Object_Detect\dataset\images"

# # Lặp qua tất cả các tệp trong thư mục
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         # Đọc hình ảnh gốc
#         img_path = os.path.join(input_folder, filename)
#         img = cv2.imread(img_path)

#         # Chuyển đổi sang dạng grayscale
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Lưu hình ảnh mới
#         output_path = os.path.join(input_folder, filename)
#         cv2.imwrite(output_path, gray_img)

# print("Hoàn thành chuyển đổi hình ảnh sang dạng grayscale.")

# import os

# # Đường dẫn đến thư mục chứa các tệp văn bản
# folder_path = r'E:\PROJECT_IN_WISDOM\Object_Detect\dataset3\labels'

# # Tạo một từ điển để lưu trữ số lượng dòng cho mỗi kí tự đầu
# line_counts = {str(i): 0 for i in range(8)}

# # Duyệt qua các tệp văn bản trong thư mục
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r") as file:
#             for line in file:
#                 # Loại bỏ khoảng trắng và ký tự newline từ đầu dòng
#                 cleaned_line = line.strip()
#                 if cleaned_line and cleaned_line[0] in "01234567":
#                     line_counts[cleaned_line[0]] += 1

# # In số lượng dòng cho mỗi kí tự đầu
# for digit, count in line_counts.items():
#     print(f"Lines starting with {digit}: {count} lines")

# import os

# # Đường dẫn đến thư mục chứa các file txt
# folder1_path = r"E:\PROJECT_IN_WISDOM\Object_Detect\dataset3\labels"  # Thay đổi thành đường dẫn thư mục chứa các file txt
# folder2_path = r"E:\PROJECT_IN_WISDOM\Object_Detect\datasix\labels"  # Thay đổi thành đường dẫn thư mục đích

# # Duyệt qua tất cả các tệp trong thư mục
# for filename in os.listdir(folder1_path):
#     if filename.endswith(".txt"):
#         input_file_path = os.path.join(folder1_path, filename)
#         output_file_path = os.path.join(folder2_path, filename)

#         # Mở tệp đầu vào để đọc
#         with open(input_file_path, 'r') as input_file:
#             # Đọc từng dòng trong tệp
#             for line in input_file:
#                 if line.startswith(("0", "1", "2", "3","4","5")):
#                     # Tạo tệp đầu ra nếu chưa tồn tại
#                     if not os.path.exists(output_file_path):
#                         open(output_file_path, 'w').close()
                    
#                     # Ghi dòng vào tệp đầu ra
#                     with open(output_file_path, 'a') as output_file:
#                         output_file.write(line)

import os

def get_common_filenames(images_dir, labels_dir):
    image_files = set([os.path.splitext(file)[0] for file in os.listdir(images_dir) if file.endswith('.jpg')])
    label_files = set([os.path.splitext(file)[0] for file in os.listdir(labels_dir) if file.endswith('.txt')])
    common_files = image_files.intersection(label_files)
    return common_files

def keep_common_files(images_dir, labels_dir):
    common_files = get_common_filenames(images_dir, labels_dir)

    for image_file in os.listdir(images_dir):
        file_name, file_ext = os.path.splitext(image_file)
        if file_ext == '.jpg' and file_name not in common_files:
            os.remove(os.path.join(images_dir, image_file))

    for label_file in os.listdir(labels_dir):
        file_name, file_ext = os.path.splitext(label_file)
        if file_ext == '.txt' and file_name not in common_files:
            os.remove(os.path.join(labels_dir, label_file))

def main():
    data_directory = "E:\\PROJECT_IN_WISDOM\\Object_Detect\\datasix"   # Thay đổi đường dẫn tới thư mục chứa data ở đây
    images_directory = os.path.join(data_directory, "images")
    labels_directory = os.path.join(data_directory, "labels")

    keep_common_files(images_directory, labels_directory)

if __name__ == "__main__":
    main()

