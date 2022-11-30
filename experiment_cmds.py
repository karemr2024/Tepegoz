import collections
import shutil
import sys
from PIL import Image
import PySpin
import time
import myspincam
import pulse_generator as pulser
import os
import scipy.io as sio
import numpy as np
import PWM_Acquisition
import moveFiles
import queue
import collections
import threading
import imageio


def pulse_for_thread(led_num):
    """

    :param led_num: 1: red, 2: green, 3: blue
    :return: return True if led is on False if LED is off.
    """
    if led_num == 0:
        pulser.one_led_pulse(led_num)
        return False
        # return True
    elif led_num == 1:
        pulser.one_led_pulse(led_num)
        return True
    elif led_num == 2:
        pulser.one_led_pulse(led_num)
        return True
    elif led_num == 3:
        pulser.one_led_pulse(led_num)
        return True
    elif led_num == 4:
        pulser.one_led_pulse(led_num)
        return True


def take_img_kill_led(cam: PySpin.CameraPtr, initter, ledon):
    """
    Makes sure LED is on for image to be taken. Takes image as a dictionary containing the
    image_array, image_timestamp, image_ptr. Make sure image is taken before led is turned off.

    :param cam: PySpin camera pointer.
    :param initter: Pre-allocated array according to image size. Use myspincam.initialise_img_arrays(cam).
    :param ledon: True if led is on False if led is off.
    :return: img is an image dictionary containing the image_array, image_timestamp, image_ptr.
    """
    if ledon:
        img, complete = myspincam.get_one_image_array(cam, initter)
        if complete:
            pulse_for_thread(0)
            return img


def pulsing_images_w_pulser(cam: PySpin.CameraPtr, imnum: int):
    """
    This function cycles through the LEDs and takes a picture for each colour.

    :param cam: PySpin camera pointer.
    :param imnum: Number of desired images.
    :return: Dictionary of image lists
    """
    img_dict = dict(red=imnum * [None], green=imnum * [None], blue=imnum * [None])  # , dark=imnum * [None])
    initter = myspincam.initialise_img_arrays(cam)
    cam.BeginAcquisition()
    # if cam.EventFrameEnd: #trying out EREN
    # if cam.EventFrameStart: #trying out EREN
    for i in range(imnum):
        # ledoff = pulse_for_thread(4)
        # img_dict['dark'][i] = take_img_kill_led(cam, initter, ledoff)
        ledon1 = pulse_for_thread(1)
        img_dict['red'][i] = take_img_kill_led(cam, initter, ledon1)
        ledon2 = pulse_for_thread(2)
        img_dict['green'][i] = take_img_kill_led(cam, initter, ledon2)
        ledon3 = pulse_for_thread(3)
        img_dict['blue'][i] = take_img_kill_led(cam, initter, ledon3)
    cam.EndAcquisition()
    return img_dict


def pulsing_images_queued(cam: PySpin.CameraPtr, imnum: int, imnam: str):
    """
    This function allows for taking images without straining the ram.
    The current drawback is that it limits the acquisition FPS at around 30
    with the current hardware.

    :param cam: PySpin camera pointer
    :param imnum: Number of desired images
    :param imnam: Desired image name
    """
    os.mkdir("C:\Tepegoz\Images" + "\_" + imnam)
    q = queue.Queue()
    q.maxsize = 100

    def queued_image_saver():
        while True:
            item = q.get()
            print(f'Working on {item}')
            sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam + "\R_" + imnam + str(item['r']['timestamp']) + ".mat"),
                        {'array': item['r']['image_array']})
            sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam + "\G_" + imnam + str(item['g']['timestamp']) + ".mat"),
                        {'array': item['g']['image_array']})
            sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam + "\B_" + imnam + str(item['b']['timestamp']) + ".mat"),
                        {'array': item['b']['image_array']})
            print(f'Finished {item}')
            q.task_done()

    t = threading.Thread(target=queued_image_saver, daemon=False)  # daemon or non daemon
    t.start()
    initter = myspincam.initialise_img_arrays(cam)
    imlist = imnum * [dict(r=np.zeros_like(initter), g=np.zeros_like(initter), b=np.zeros_like(initter))]

    cam.BeginAcquisition()
    for i in range(imnum):
        ledon1 = pulse_for_thread(1)
        imlist[i]['r'] = take_img_kill_led(cam, initter, ledon1)
        ledon2 = pulse_for_thread(2)
        imlist[i]['g'] = take_img_kill_led(cam, initter, ledon2)
        ledon3 = pulse_for_thread(3)
        imlist[i]['b'] = take_img_kill_led(cam, initter, ledon3)
        q.put(imlist[i])
    cam.EndAcquisition()

    q.join()
    t.join()

def pulsing_images_dequeued(cam: PySpin.CameraPtr, imnum: int, imnam: str): #has been switched between queue & deque
    """
    This function allows for taking images without straining the ram.
    The current drawback is that it limits the acquisition FPS at around 30
    with the current hardware.

    :param cam: PySpin camera pointer
    :param imnum: Number of desired images
    :param imnam: Desired image name
    """
    # os.mkdir("C:\Tepegoz\Images" + "\_" + imnam)
    os.mkdir("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam)
    # qr = collections.deque()
    # qr.maxsize = 100
    qr = queue.Queue()
    # qg = collections.deque()
    # qg.maxsize = 100
    qg = queue.Queue()
    # qb = collections.deque()
    # qb.maxsize = 100
    qb = queue.Queue()

    def dequeued_r_image_saver():
        # os.mkdir("C:\Tepegoz\Images" + "\_" + imnam + "\R_" + imnam)
        os.mkdir("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\R_" + imnam)
        while True:
            # item = qr.popleft()
            item = qr.get()
            print(f'Working on {item}')
            sio.savemat(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\R_" + imnam + "\R_" + imnam + str(item['r']['timestamp']) + ".mat"),
                        {'array': item['r']['image_array']})
            # imageio.imwrite(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\R_" + imnam + "\R_" + imnam + str(
            #     item['r']['timestamp']) + ".tif"), item['r']['image_array'], format='tiff')
            print(f'Finished {item}')
            del item


    def dequeued_g_image_saver():
        # os.mkdir("C:\Tepegoz\Images" + "\_" + imnam + "\G_" + imnam)
        os.mkdir("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\G_" + imnam)
        while True:
            # item = qg.popleft()
            item = qg.get()
            print(f'Working on {item}')
            sio.savemat(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\G_" + imnam + "\G_" + imnam + str(item['g']['timestamp']) + ".mat"),
                        {'array': item['g']['image_array']})
            # imageio.imwrite(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\G_" + imnam + "\G_" + imnam + str(
            #     item['g']['timestamp']) + ".tif"), item['g']['image_array'], format='tiff')
            print(f'Finished {item}')
            del item


    def dequeued_b_image_saver():
        # os.mkdir("C:\Tepegoz\Images" + "\_" + imnam + "\B_" + imnam)
        os.mkdir("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\B_" + imnam)
        while True:
            # item = qb.popleft()
            item = qb.get()  # or pop?
            print(f'Working on {item}')
            sio.savemat(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\B_" + imnam + "\B_" + imnam + str(item['b']['timestamp']) + ".mat"),
                        {'array': item['b']['image_array']})
            # imageio.imwrite(str("C:\Tepegoz\iRiS_Kinetics_Github\experiments\Images" + "\_" + imnam + "\B_" + imnam + "\B_" + imnam + str(
            #     item['b']['timestamp']) + ".tif"), item['b']['image_array'], format='tiff')
            print(f'Finished {item}')
            del item

    tr = threading.Thread(target=dequeued_r_image_saver, daemon=False)  # daemon or non daemon
    tg = threading.Thread(target=dequeued_g_image_saver, daemon=False)  # daemon or non daemon
    tb = threading.Thread(target=dequeued_b_image_saver, daemon=False)  # daemon or non daemon
    tr.start()
    tg.start()
    tb.start()

    initter = myspincam.initialise_img_arrays(cam)
    rimlist = imnum * [dict(r=np.zeros_like(initter))]
    gimlist = imnum * [dict(g=np.zeros_like(initter))]
    bimlist = imnum * [dict(b=np.zeros_like(initter))]


    cam.BeginAcquisition()
    for i in range(imnum):
        ledon1 = pulse_for_thread(1)
        rimlist[i]['r'] = take_img_kill_led(cam, initter, ledon1)
        qr.put_nowait(rimlist[i]) # put_nowait() or put()
        ledon2 = pulse_for_thread(2)
        gimlist[i]['g'] = take_img_kill_led(cam, initter, ledon2)
        qg.put_nowait(gimlist[i]) # put_nowait() or put()
        ledon3 = pulse_for_thread(3)
        bimlist[i]['b'] = take_img_kill_led(cam, initter, ledon3)
        qb.put_nowait(bimlist[i]) # put_nowait() or put()
    cam.EndAcquisition()

    qr.join()
    qg.join()
    qb.join()
    tr.join()
    tg.join()
    tb.join()
    del qr
    del qg
    del qb
    del tr
    del tg
    del tb
    pass

def pulse_fast_and_save_pulser(cam, imnum):
    """
    :param cam: PySpin camera pointer.
    :param imnum: Number of desired images.
    :return: PIL images from arrays.

    Each key in the imgs dictionary is a colour. A list of images
    are assigned to each colour, in the order they are taken.
    with one PySPin image dictionary assigned to the keys red, green, blue.
    You may visualise this nested structure as such:

    imgs: dict
    imgs = {'red':[{'image_array':[], 'image_timestamp': int, 'image_ptr': PySpin Image Pointer}, ...],
            'green':[{'image_array':[], 'image_timestamp': int, 'image_ptr': PySpin Image Pointer}, ...],
            'blue':[{'image_array':[], 'image_timestamp': int, 'image_ptr': PySpin Image Pointer}, ...]}
    """
    t1_start = float(time.clock())
    imgs = pulsing_images_w_pulser(cam, imnum)
    t1_stop = float(time.clock())
    print("pulse start: %f" % t1_start)
    print("pulse stop: %f" % t1_stop)
    print("pulse elapsed: %f" % (t1_stop - t1_start))
    print("Saving Images as Arrays...")
    r = []
    g = []
    b = []
    for i in range(len(imgs['red'])):
        # imgs[i]['red']['image_ptr'].Save(str(i) + "_R_" + str(imnam) + ".tif")
        # Keep in mind that imnam has been erased as input argument.
        r.append(Image.fromarray(imgs['red'][i]['image_array']))
        # print(imgs['red'][i]['timestamp'])

        # imgs[i]['green']['image_ptr'].Save(str(i) + "_G_" + str(imnam) + ".tif")
        # Keep in mind that imnam has been erased as input argument.
        g.append(Image.fromarray(imgs['green'][i]['image_array']))
        # print(imgs['green'][i]['timestamp'])

        # imgs[i]['blue']['image_ptr'].Save(str(i) + "_B_" + str(imnam) + ".tif")
        # Keep in mind that imnam has been erased as input argument.
        b.append(Image.fromarray(imgs['blue'][i]['image_array']))
        # print(imgs['blue'][i]['timestamp'])

    return r, g, b


def save_as_tiff(r: list, g: list, b: list, imnam: str):
    """
    A folder is created in the desired image name.
    Images are saved as tiff stacks for each colour
    with the number of elements chosen in the experiment.
    This process takes a while.

    :param r: List in which red PIL images are saved.
    :param g: List in which green PIL images are saved.
    :param b: List in which blue PIL images are saved.
    :param imnam: Desired image name.
    """
    print("Saving Images as Tiff Stacks ...")
    t1_start = float(time.clock())
    os.mkdir("C:\Tepegoz\Images" + "\_" + imnam)
    r[0].save("C:\Tepegoz\Images" + "\_" + imnam + "\_" + imnam + "_R.tif",
              compression="tiff_deflate", save_all=True,
              append_images=r[1:])
    g[0].save("C:\Tepegoz\Images" + "\_" + imnam + "\_" + imnam + "_G.tif",
              compression="tiff_deflate", save_all=True,
              append_images=g[1:])
    b[0].save("C:\Tepegoz\Images" + "\_" + imnam + "\_" + imnam + "_B.tif",
              compression="tiff_deflate", save_all=True,
              append_images=b[1:])
    t1_stop = float(time.clock())
    print("pulse start: %f" % t1_start)
    print("pulse stop: %f" % t1_stop)
    print("pulse elapsed: %f" % (t1_stop - t1_start))
    print("Images Successfully Saved as Tiff Stacks.\n")
    pass


def save_as_mat(img_dict: dict, imnam_ipt: str):
    os.mkdir("C:\Tepegoz\Images" + "\_" + imnam_ipt)
    r = {}
    g = {}
    b = {}
    # d = {}
    for i in range(len(img_dict['red'])):
        # d[imnam_ipt + "_" + str(i + 1) + "_D_array"] = np.array(img_dict['dark'][i]['image_array'])
        # d[imnam_ipt + "_" + str(i + 1) + "_D_timestamp"] = img_dict['dark'][i]['timestamp']
        r[imnam_ipt + "_" + str(i + 1) + "_R_array"] = np.array(img_dict['red'][i]['image_array'])
        r[imnam_ipt + "_" + str(i + 1) + "_R_timestamp"] = img_dict['red'][i]['timestamp']
        g[imnam_ipt + "_" + str(i + 1) + "_G_array"] = np.array(img_dict['green'][i]['image_array'])
        g[imnam_ipt + "_" + str(i + 1) + "_G_timestamp"] = img_dict['green'][i]['timestamp']
        b[imnam_ipt + "_" + str(i + 1) + "_B_array"] = np.array(img_dict['blue'][i]['image_array'])
        b[imnam_ipt + "_" + str(i + 1) + "_B_timestamp"] = img_dict['blue'][i]['timestamp']

    # sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\D_" + imnam_ipt + ".mat"), d)
    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\R_" + imnam_ipt + ".mat"), r)
    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\G_" + imnam_ipt + ".mat"), g)
    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\B_" + imnam_ipt + ".mat"), b)


def save_as_mat_rollingavg(img_dict: dict, imnam_ipt: str):
    os.mkdir("C:\Tepegoz\Images" + "\_" + imnam_ipt)
    r = len(np.array(img_dict['red'])) * [np.empty_like(img_dict['red'][2]['image_array'])]
    g = len(np.array(img_dict['green'])) * [np.empty_like(img_dict['green'][2]['image_array'])]
    b = len(np.array(img_dict['blue'])) * [np.empty_like(img_dict['blue'][2]['image_array'])]
    r_dict = {}
    g_dict = {}
    b_dict = {}

    for i in range(len(img_dict['red'])):
        r[i] = np.array(img_dict['red'][i]['image_array'])
        g[i] = np.array(img_dict['green'][i]['image_array'])
        b[i] = np.array(img_dict['blue'][i]['image_array'])

    n = 100
    r_avg = list([np.mean(r[i:i + n], axis=0, dtype=np.float64)] for i in range(0, len(r), n))
    g_avg = list([np.mean(g[i:i + n], axis=0, dtype=np.float64)] for i in range(0, len(g), n))
    b_avg = list([np.mean(b[i:i + n], axis=0, dtype=np.float64)] for i in range(0, len(b), n))

    for i in range(len(r_avg)):
        r_dict[imnam_ipt + "_" + str(i + 1) + "_R_array"] = r_avg[i]
        g_dict[imnam_ipt + "_" + str(i + 1) + "_G_array"] = g_avg[i]
        b_dict[imnam_ipt + "_" + str(i + 1) + "_B_array"] = b_avg[i]

    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\R_" + imnam_ipt + ".mat"), r_dict)
    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\G_" + imnam_ipt + ".mat"), g_dict)
    sio.savemat(str("C:\Tepegoz\Images" + "\_" + imnam_ipt + "\B_" + imnam_ipt + ".mat"), b_dict)


def apply_default_experiment_settings(cam):
    """
    Apply default settings. Values are displayed on UI by default.

    :param cam: PySpin camera pointer.
    """
    myspincam.get_cam_and_init(cam)
    myspincam.set_acquisition_mode_continuous(cam)
    # myspincam.set_acquisition_mode_singleframe(cam)
    # myspincam.set_trigger(cam)
    myspincam.set_frame_rate(cam, 1)
    myspincam.disable_auto_exp(cam)
    myspincam.disable_auto_gain(cam)
    myspincam.set_exposure(cam, 4000)
    myspincam.set_gain(cam, 0)
    myspincam.set_gamma(cam, 1)
    myspincam.set_pixel_format(cam)
    myspincam.set_cam_buffer(cam)
    # myspincam.configure_trigger(cam, "SOFTWARE")
    pulser.connect("5")


def apply_custom_experiment_setttings(cam, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    """
    Apply experiment settings according to user input in UI.
    Currently, trying out different buffer handling modes...

    :param cam: PySpin camera pointer.
    :param gain_ipt:
    :param fps_ipt: Suggested FPS values between 0 & 80. Working on optimisation.
    :param exposure_ipt: Exposure values given in microseconds.
    """
    myspincam.get_cam_and_init(cam)
    myspincam.set_acquisition_mode_continuous(cam)
    myspincam.set_frame_rate(cam, fps_ipt)
    myspincam.disable_auto_exp(cam)
    myspincam.disable_auto_gain(cam)
    myspincam.set_exposure(cam, exposure_ipt)
    myspincam.set_gain(cam, gain_ipt)
    myspincam.set_gamma(cam, 1)
    myspincam.set_pixel_format(cam)
    myspincam.set_cam_buffer(cam)
    # myspincam.configure_trigger(cam, "SOFTWARE")
    pulser.connect("5")


def run_experiment_tiff(imnum_ipt: int, imnam_ipt: str, setting: str, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    print("Starting experiment")
    __SYSTEM = PySpin.System.GetInstance()
    __CAM = __SYSTEM.GetCameras()[0]

    if setting == "DEFAULT":
        apply_default_experiment_settings(__CAM)
    elif setting == "CUSTOM":
        apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)

    time.sleep(3)
    r, g, b = pulse_fast_and_save_pulser(__CAM, imnum_ipt)
    save_as_tiff(r, g, b, imnam_ipt)
    pulser.one_led_pulse(0)
    time.sleep(1)
    __CAM.DeInit()
    pass


def run_experiment_mat(imnum_ipt: int, imnam_ipt: str, setting: str, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    print("Starting experiment")
    __SYSTEM = PySpin.System.GetInstance()
    __CAM = __SYSTEM.GetCameras()[0]

    if setting == "DEFAULT":
        apply_default_experiment_settings(__CAM)
    elif setting == "CUSTOM":
        apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)

    time.sleep(3)
    img_dict = pulsing_images_w_pulser(__CAM, imnum_ipt)
    save_as_mat(img_dict, imnam_ipt)
    pulser.one_led_pulse(0)
    time.sleep(1)
    __CAM.DeInit()
    __SYSTEM.ReleaseInstance()


def run_experiment_rollingmatmean(imnum_ipt: int, imnam_ipt: str, setting: str, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    print("Starting experiment")
    __SYSTEM = PySpin.System.GetInstance()
    __CAM = __SYSTEM.GetCameras()[0]

    if setting == "DEFAULT":
        apply_default_experiment_settings(__CAM)
    elif setting == "CUSTOM":
        apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)

    time.sleep(3)
    numba = int(imnum_ipt/1000)
    leftova = imnum_ipt % 1000
    if numba > 1:
        a = 1
        for i in range(numba):
            # pulsing_images_queued(__CAM, 1000, str(i+1) + imnam_ipt)
            pulsing_images_dequeued(__CAM, 1000, str(str(i + 1) + "_" + imnam_ipt))
            a = a+1
            __CAM.DeInit()
            __SYSTEM.ReleaseInstance()
            pulser.one_led_pulse(4)
            pulser.close()
            del __CAM
            del __SYSTEM
            __SYSTEM = PySpin.System.GetInstance()
            print("stuck")
            __CAM = __SYSTEM.GetCameras()[0]
            time.sleep(60)
            if setting == "DEFAULT":
                apply_default_experiment_settings(__CAM)
            elif setting == "CUSTOM":
                apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)
        # pulsing_images_queued(__CAM, leftova, str(a) + imnam_ipt)
        pulsing_images_dequeued(__CAM, leftova, str(a) + imnam_ipt)
    elif numba <= 1:
        # pulsing_images_queued(__CAM, imnum_ipt, imnam_ipt)
        pulsing_images_dequeued(__CAM, imnum_ipt, imnam_ipt)

    # pulsing_images_ram_relief(__CAM, imnum_ipt)
    # pulsing_images_buffering(__CAM, imnum_ipt, imnam_ipt)
    # moveFiles.np2mat()
    # nps_to_avgmats(imnum_ipt, imnam_ipt)
    # img_dict = pulsing_images_w_pulser(__CAM, imnum_ipt)
    # save_as_mat_rollingavg(img_dict, imnam_ipt)
    pulser.one_led_pulse(0)
    time.sleep(1)
    __CAM.DeInit()
    __SYSTEM.ReleaseInstance()


def run_experiment_optimised(imnum_ipt: int, imnam_ipt: str, setting: str, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    print("Starting experiment")
    __SYSTEM = PySpin.System.GetInstance()
    __CAM = __SYSTEM.GetCameras()[0]

    if setting == "DEFAULT":
        apply_default_experiment_settings(__CAM)
    elif setting == "CUSTOM":
        apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)

    time.sleep(2)
    pulsing_images_dequeued(__CAM, imnum_ipt, imnam_ipt)
    # pulser.one_led_pulse(4)
    pulser.one_led_pulse(0)
    time.sleep(1)
    __CAM.DeInit()
    __SYSTEM.ReleaseInstance()
    del __CAM
    del __SYSTEM
    pass


def run_experiment_nogui(imnum_ipt: int, imnam_ipt: str, setting: str, gain_ipt: float, fps_ipt: int, exposure_ipt: int):
    print("Starting experiment")
    __SYSTEM = PySpin.System.GetInstance()
    __CAM = __SYSTEM.GetCameras()[0]

    if setting == "DEFAULT":
        apply_default_experiment_settings(__CAM)
    elif setting == "CUSTOM":
        apply_custom_experiment_setttings(__CAM, gain_ipt, fps_ipt, exposure_ipt)

    time.sleep(3)
    pulsing_images_dequeued(__CAM, imnum_ipt, imnam_ipt)
    pulser.one_led_pulse(0)
    pulser.one_led_pulse(4)
    time.sleep(1)
    __CAM.DeInit()
    __SYSTEM.ReleaseInstance()
    del __CAM
    del __SYSTEM
    pass
