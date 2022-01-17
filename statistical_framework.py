#Importation des librairies
import numpy as np
import scipy as sp
from scipy.spatial import distance
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import os


########################################################################################################################
#Définition des fonctions


def get_raw_coordinates(path_file):
    file = open(path_file, 'r')
    raw_coordinates = []
    for line in file:
        coordinate_buffer = line.split(",")
        raw_coordinates.append((float(coordinate_buffer[0]), float(coordinate_buffer[1])))
    file.close()
    return raw_coordinates


def get_image_size(image_file_name, image_file_name_extension):
    images_base_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/ALLSTIMULI/"
    image_file = Image.open(images_base_path + image_file_name + '.' + image_file_name_extension, 'r')
    width, height = image_file.size
    return width, height


def draw_gaze_path(image_file_name, image_file_name_extension, dir_name):
    images_base_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/ALLSTIMULI/"
    dir_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/DATACSV/"
    raw_coordinates = get_raw_coordinates(dir_path + dir_name + "/" + image_file_name)
    coordinates = delete_wrong_values(raw_coordinates, get_image_size(image_file_name, image_file_name_extension='jpeg'))
    image_file = Image.open(images_base_path + image_file_name + '.' + image_file_name_extension, 'r')
    image_draw = ImageDraw.Draw(image_file, "RGBA")
    image_draw.line(coordinates,  fill='red', width=2, joint='curve')
    image_file.show()



def delete_wrong_values(coordinates, image_size):
    # Verifier height widht avec image_size et coord
    image_width = image_size[0]
    image_height = image_size[1]
    corrected_coordinates = [0] * len(coordinates)
    for it in range(len(coordinates)):
        if np.isnan(coordinates[it][0]) or np.isnan(coordinates[it][1]) is True:
            coordinates[it] = (-1000., -1000.)
    i = 0
    coord_buffer = coordinates[0]
    for coord in coordinates:
        if coord[0] > 0 and coord[0] < image_width and coord[1] > 0 and coord[1] < image_height or distance.euclidean(
                coord, coord_buffer) <= 100:
            corrected_coordinates[i] = coord
            i += 1
            coord_buffer = coord
    corrected_coordinates = corrected_coordinates[0: i - 1]
    return corrected_coordinates


def coordinate2movement(coordinates):
    movement_process = np.zeros((len(coordinates), 1))  # movement vecteur colonne
    for i in range(1, len(coordinates)):
        movement_process[i - 1] = np.sqrt((coordinates[i][0] - coordinates[i - 1][0]) ** 2 + (
                    coordinates[i][1] - coordinates[i - 1][1]) ** 2)
    return movement_process


def cumsum_process(movement_process):  # A finir sur le calcul de f
    # calcul de f
    f = 2.5
    cumsum_process = [0] * len(movement_process)
    index_cumsum_process = [0] * len(movement_process)
    St = 0
    index = 0
    it = 0
    for movement in movement_process:
        if movement > 0:
            St += movement
            cumsum_process[it] = float(max(0., St - it * f))
            index_cumsum_process[it] = index
            it += 1
        index += 1
    cumsum_process = cumsum_process[0: it - 1]
    index_cumsum_process = index_cumsum_process[0: it - 1]
    return cumsum_process, index_cumsum_process


def rebuilted_movement_process(movement_process, index_cumsum_process):
    rebuild_movement_process = [0.] * len(movement_process)
    for index in index_cumsum_process:
        rebuild_movement_process[index] = float(movement_process[index])
    return rebuild_movement_process


def bin_seg_estimand(s, e, process, b):
    if len(process) >= e - s and b - s < e - s and b - s > 0:
        return abs(np.sqrt((e - b) / ((e - s + 1) * (b - s + 1))) * np.sum(np.array(process[s:b + 1])) - np.sqrt(
            (b - s + 1) / ((e - s + 1) * (e - b))) * np.sum(np.array(process[b + 1:e + 1])))


def max_bin_seg_estimand(s, e, process):
    # definition of a base of acceptable values
    delta_t = 10  # saccades cannot be in a delta length interval
    B = [b for b in range(s + 1 + delta_t, e - delta_t, )]
    max_estimand = 0.
    argmax_estimand = 0
    it = s + 1 + delta_t
    for b in B:
        estimand_buffer = bin_seg_estimand(s, e, process, b)
        it += 1
        if estimand_buffer > max_estimand:
            max_estimand = estimand_buffer
            argmax_estimand = it
    return max_estimand, argmax_estimand


def bin_segmentation(s, e, process, threshold, set_estimated_change_point):
    delta_t = 10
    if e - s > 2*delta_t + 1: #initialement e - s > 1
        min_process_se = min(process[s: e+1])
        corrected_process = [p - min_process_se for p in process]
        max_estimand, argmax_estimand = max_bin_seg_estimand(s, e, corrected_process)
        if max_estimand > threshold:
            bin_segmentation(s, argmax_estimand, process, threshold, set_estimated_change_point)
            bin_segmentation(argmax_estimand + 1, e, process, threshold, set_estimated_change_point)
            set_estimated_change_point.append(argmax_estimand)


def get_segmentation(process, threshold):
    set_estimated_change_point = []
    bin_segmentation(1, len(process), process, threshold, set_estimated_change_point)
    return sorted(set_estimated_change_point)


def get_rebuilted_segmentation(cumsum_process, cumsum_process_index, threshold):
    set_change_point = get_segmentation(cumsum_process, threshold)
    return [cumsum_process_index[b] for b in set_change_point]


def plot_discriminated_saccades(movement_process, threshold):
    delta_t = 10
    t_linspace = np.arange(len(movement_process))
    # Use of previous definited function to get well localised estimated change points
    cumsum, cumsum_index = cumsum_process(movement_process)
    rebuilted_process = rebuilted_movement_process(movement_process, cumsum_index)
    set_estimated_change_point = get_rebuilted_segmentation(cumsum, cumsum_index, threshold)
    # Definition of lists to plot
    saccade_regime = [np.nan] * len(rebuilted_process)
    microsaccade_regime = [np.nan] * len(rebuilted_process)
    lower_bound_buffer = 0
    for estimated_change_point in set_estimated_change_point:
        saccade_regime[estimated_change_point - delta_t: estimated_change_point + delta_t + 1] = rebuilted_process[
                                                                                                 estimated_change_point - delta_t: estimated_change_point + delta_t + 1]
        microsaccade_regime[lower_bound_buffer: estimated_change_point - delta_t + 1] = rebuilted_process[
                                                                                        lower_bound_buffer: estimated_change_point - delta_t + 1]
        lower_bound_buffer = estimated_change_point + delta_t
    microsaccade_regime[lower_bound_buffer: len(rebuilted_process)] = rebuilted_process[
                                                                      lower_bound_buffer: len(rebuilted_process)]
    plt.figure()
    plt.plot(t_linspace, microsaccade_regime)
    plt.plot(t_linspace, saccade_regime, 'r')
    plt.show()


def get_discriminated_saccades(movement_process, threshold):
    delta_t = 10
    # Use of previous definited function to get well localised estimated change points
    cumsum, cumsum_index = cumsum_process(movement_process)
    set_estimated_change_point = get_rebuilted_segmentation(cumsum, cumsum_index, threshold)
    saccades_intervals = [(b - delta_t, b + delta_t) for b in set_estimated_change_point]
    return saccades_intervals


def get_discriminated_microsaccades(movement_process, saccades_intervals):
    delta_t = 10
    if len(saccades_intervals) > 1:
        microsaccades_intervals = []
        for i in range(0, len(saccades_intervals)):
            if i == 0:
                microsaccades_intervals.append((0, saccades_intervals[i][0] - 1))
            else:
                microsaccades_intervals.append((saccades_intervals[i - 1][1] + 1, saccades_intervals[i][0] - 1))
        microsaccades_intervals.append((saccades_intervals[len(saccades_intervals) - 1][1] + 1, len(movement_process)))
        # Finding of indices of impossible intervals
        deleted_intervals_index = []
        it = 0
        for interval in microsaccades_intervals:
            if interval[1] - interval[0] < delta_t + 1 or interval[1] - interval[0] < 0:
                deleted_intervals_index.append(it)
            it += 1
        # Deletation of impossile intervals
        if len(deleted_intervals_index) >= 1:
            deleted_intervals_index = sorted(deleted_intervals_index)
            for j in range(len(deleted_intervals_index)):
                del microsaccades_intervals[deleted_intervals_index[j]]
                deleted_intervals_index = [index - 1 for index in deleted_intervals_index]
    else:
        microsaccades_intervals = [(0, len(movement_process))]
    return microsaccades_intervals


def get_barycenter_points_of_interest(microsaccades_intervals, coordinates):
    barycenter_points_of_interest = [0] * len(microsaccades_intervals)
    it = 0
    for interval in microsaccades_intervals:
        x_star = 0.
        y_star = 0.
        for i in range(interval[0], interval[
            1]):  # On prend le parti de prendre un point de moins (celui juste avant la saccade) car on repasse des accroissements aux coordonnées
            x_star += coordinates[i][0]
            y_star += coordinates[i][1]
        x_star = int(round(x_star / (interval[1] - interval[0])))
        y_star = int(round(y_star / (interval[1] - interval[0])))
        barycenter_points_of_interest[it] = (x_star, y_star)
        it += 1
    # Deletation of barycenters of false positive microsaccade, i.e barycenters that are very close to the previous one
    deleted_index = []
    for i in range(1, len(barycenter_points_of_interest)):
        if barycenter_points_of_interest[i] == (0, 0):
            deleted_index.append(i)
            # del barycenter_points_of_interest[i]
        if distance.euclidean(barycenter_points_of_interest[i - 1], barycenter_points_of_interest[i]) < 25:
            x_star_buffer = int(
                round((barycenter_points_of_interest[i - 1][0] + barycenter_points_of_interest[i][0]) / 2))
            y_star_buffer = int(
                round((barycenter_points_of_interest[i - 1][1] + barycenter_points_of_interest[i][1]) / 2))
            barycenter_points_of_interest[i - 1] = (x_star_buffer, y_star_buffer)
            deleted_index.append(i)
            # del barycenter_points_of_interest[i]
    if len(deleted_index) >= 1:
        deleted_index = sorted(deleted_index)
        for j in range(len(deleted_index)):
            del barycenter_points_of_interest[deleted_index[j]]
            deleted_index = [index - 1 for index in deleted_index]
    return barycenter_points_of_interest


def deleted_list_element_from_a_list_index(a_list, index_list):
    for j in range(index_list):
        del a_list[index_list[j]]
        index_list = [index - 1 for index in index_list]
    return a_list


def from_file_plot_discriminated_saccades(file_name, dir_name, threshold):
    dir_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/DATACSV/"
    raw_coordinates = get_raw_coordinates(dir_path + dir_name + "/" + file_name)
    coordinates = delete_wrong_values(raw_coordinates, get_image_size(file_name, image_file_name_extension='jpeg'))
    plot_discriminated_saccades(coordinate2movement(coordinates), threshold)


def from_file_get_barycenters_points_of_interest(file_name, dir_name, threshold):
    dir_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/DATACSV/"
    raw_coordinates = get_raw_coordinates(dir_path + dir_name + "/" + file_name)
    coordinates = delete_wrong_values(raw_coordinates, get_image_size(file_name, image_file_name_extension='jpeg'))
    movement_process = coordinate2movement(coordinates)
    saccades_intervals = get_discriminated_saccades(movement_process, threshold)
    microsaccades_intervals = get_discriminated_microsaccades(movement_process, saccades_intervals)
    barycenters = get_barycenter_points_of_interest(microsaccades_intervals, coordinates)
    return barycenters


def draw_barycenters_points_of_interest(file_name, image_file_name_extension, dir_name, threshold):
    barycenters = from_file_get_barycenters_points_of_interest(file_name, dir_name, threshold)
    images_base_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/ALLSTIMULI/"
    dir_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/DATACSV/"
    raw_coordinates = get_raw_coordinates(dir_path + dir_name + "/" + file_name)
    coordinates = delete_wrong_values(raw_coordinates,
                                      get_image_size(file_name, image_file_name_extension='jpeg'))
    image_file = Image.open(images_base_path + file_name + '.' + image_file_name_extension, 'r')
    image_draw = ImageDraw.Draw(image_file, "RGBA")
    image_draw.line(coordinates, fill='red', width=2, joint='curve')
    r = 20
    i = 1
    for b in barycenters:
        image_draw.arc((b[0] - r, b[1] - r, b[0] + r, b[1] + r), start=0, end=360, fill='yellow')
        text = "%d" % i
        image_draw.text((b[0], b[1]), text)
        i += 1
    image_file.show()


########################################################################################################################
#Création de la base
"""

data_path = "C:/Users/Linguini/Documents/Centrale_Marseille/3A/Digital.e/Projet_Computer_Vision/EyetrackingDatabaseFolder/"
data_base_path_to_save = data_path + "DATABASE/"
eyetracking_data_path = data_path + "DATACSV/"
guinea_pigs_names = os.listdir(eyetracking_data_path)
for name in guinea_pigs_names:
    buffer_path = eyetracking_data_path + name + "/"
    buffer_os_list = os.listdir(buffer_path)
    os.makedirs(data_base_path_to_save + name + "/")
    for file in buffer_os_list:
        file_name_buffer = file
        barycenters_buffer = from_file_get_barycenters_points_of_interest(file_name_buffer, name, threshold=200)
        dict_buffer = {'image_name': file_name_buffer, 'barycenters': barycenters_buffer}
        outfile = open(data_base_path_to_save + name + "/" + file_name_buffer, 'wb')
        pickle.dump(dict_buffer, outfile)
        outfile.close()

"""

from_file_plot_discriminated_saccades('i167462665', 'CNG', threshold=200)
draw_barycenters_points_of_interest('i167462665', 'jpeg', 'CNG', threshold=200)
#print(from_file_get_barycenters_points_of_interest('i167462665', 'CNG', threshold=200))

#Problème notable dans la base
#CNG i1246371431 Pas de saccade, seulement un point de fixation (fixé)
#emb i1065169436 Présence de NaN values dans les coordonnées (fixé)

###------------------------------------------------------###
