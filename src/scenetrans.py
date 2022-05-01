import random
import numpy as np
from typing import Any

# from src import augmentation
# from src import visualization

ALPHA_SMALL = 0.5
ALPHA_BIG = 2
# ALPHA_SMALL = 0.2
# ALPHA_BIG = 5

BETA = 1.1

class OneOf():
    """ 
    Select one of transformations in list to apply
    """
    def __init__(self, transforms, p: float = 0.5) -> None:
        self.p = p
        self.transforms = transforms
        self.tp = [self.transforms[i].p for i in range(len(self.transforms))]
        self.tp = [p / sum(self.tp) for p in self.tp]

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            tidx = np.random.choice(len(self.tp), p=self.tp)
            t = self.transforms[tidx]
            sample = t(sample)
        return sample


class ThisIsAObject(object):
    """ This is a object.
    """
    def __init__(self, p: float = 0.5, noprompt: bool = False) -> None:
        self.p = p
        self.pattern = "This is a {Object}." if not noprompt else "{Object}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            text = sample["text"]
            sample["text"] = self.pattern.format(Object=text)
        return sample


class ABigObject(object):
    """ A big object.
        Object get larger 2 times.
    """
    def __init__(self, p: float = 0.5, noprompt: bool = False) -> None:
        self.p = p
        self.pattern = "A big {Object}." if not noprompt else "{Object}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            text = sample["text"]
            points = sample["points"]

            # make object big
            sample["points"] = ALPHA_BIG * points

            sample["text"] = self.pattern.format(Object=text)
        return sample


class ASmallObject(object):
    """ A Small object.
        Object get smaller by 0.5 factor.
    """
    def __init__(self, p: float = 0.5, noprompt: bool = False) -> None:
        self.p = p
        self.pattern = "A small {Object}." if not noprompt else "{Object}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            text = sample["text"]
            points = sample["points"]

            # make object small
            sample["points"] = ALPHA_SMALL * points

            sample["text"] = self.pattern.format(Object=text)
        return sample


class TwoObjects(object):
    """ Two objects.
    """
    def __init__(self, p: float = 0.5, noprompt: bool = False) -> None:
        self.p = p
        self.pattern = "Two {Object}s." if not noprompt else "{Object}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            text = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]
            arglabel = np.argwhere(labels.reshape(-1,)==label)

            # select another object index randomly
            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0][0]]

            # randomly shift second object in x and y direction
            pointsB = xshift(pointsB, is_random=True)
            pointsB = zshift(pointsB, is_random=True)

            # combine objects
            points = np.vstack((pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = xzcentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(Object=text)
        return sample


class TwoCloseObjects(object):
    """ Two close objects.
    """
    def __init__(self, p: float = 0.5, noprompt: bool = False) -> None:
        self.p = p
        self.pattern = "Two close {Object}s." if not noprompt else "{Object}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            text = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]

            # randomly choose a different object 
            arglabel = np.argwhere(labels.reshape(-1,)==label)
            # select another object index randomly
            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0][0]]

            # randomly shift second object in x and y direction
            pointsB = xshift(pointsB, ratio=BETA)
            pointsB = zshift(pointsB, ratio=BETA)

            # combine objects
            points = np.vstack((pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = xzcentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(Object=text)
        return sample



class AObjectAIsCloseToObjectB(object):
    """ A Object is close to ObjectB.
    """
    def __init__(self, p: float = 0.5, within: bool = False, noprompt: bool = False) -> None:
        self.p = p
        self.within = within
        self.pattern = "A {ObjectA} is close to {ObjectB}." if not noprompt else "{ObjectA} {ObjectB}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            textA = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]

            # randomly choose a different object 
            # if within is True objectA is same as objectB else they are different
            arglabel = np.argwhere(labels.reshape(-1,)==label) if self.within else np.argwhere(labels.reshape(-1,)!=label)

            # select another object index randomly
            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0][0]]
            labelB = labels[arglabel[0][0]][0]
            textB = sample["all_class_names"][labelB]

            # randomly shift second object in x and y direction
            pointsB = xshift(pointsB, ratio=BETA)
            pointsB = zshift(pointsB, ratio=BETA)

            # combine objects
            points = np.vstack((pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = xzcentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(ObjectA=textA, ObjectB=textB)
        return sample



class ABigObjectAIsCloseToObjectB(object):
    """ A big Object is close to ObjectB.
    """
    def __init__(self, p: float = 0.5, within: bool = False, noprompt: bool = False) -> None:
        self.p = p
        self.within = within
        self.pattern = "A big {ObjectA} is close to {ObjectB}." if not noprompt else "{ObjectA} {ObjectB}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            textA = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]

            # randomly choose a different object
            # if within is True objectA is same as objectB else they are different
            arglabel = np.argwhere(labels.reshape(-1,)==label) if self.within else np.argwhere(labels.reshape(-1,)!=label)

            # select another object index randomly
            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0][0]]
            labelB = labels[arglabel[0][0]][0]
            textB = sample["all_class_names"][labelB]

            # randomly shift second object in x and y direction
            pointsB = xshift(pointsB, ratio=BETA)
            pointsB = zshift(pointsB, ratio=BETA)

            # combine objects
            points = np.vstack((ALPHA_BIG * pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = xzcentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(ObjectA=textA, ObjectB=textB)
        return sample


class ASmallObjectAIsCloseToObjectB(object):
    """ A small Object is close to ObjectB.
    """
    def __init__(self, p: float = 0.5, within: bool = False, noprompt: bool = False) -> None:
        self.p = p
        self.within = within
        self.pattern = "A small {ObjectA} is close to {ObjectB}." if not noprompt else "{ObjectA} {ObjectB}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            textA = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]

            # randomly choose a different object
            # if within is True objectA is same as objectB else they are different
            arglabel = np.argwhere(labels.reshape(-1,)==label) if self.within else np.argwhere(labels.reshape(-1,)!=label)

            # select another object index randomly
            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0][0]]
            labelB = labels[arglabel[0][0]][0]
            textB = sample["all_class_names"][labelB]

            # randomly shift second object in x and y direction
            pointsB = xshift(pointsB, ratio=BETA)
            pointsB = zshift(pointsB, ratio=BETA)

            # combine objects
            points = np.vstack((ALPHA_SMALL * pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = xzcentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(ObjectA=textA, ObjectB=textB)
        return sample


class AObjectAIsOnObjectB(object):
    """ A ObjectA is close to ObjectB.
    """
    def __init__(self, p: float = 0.5, within: bool = False, noprompt: bool = False) -> None:
        self.p = p
        self.within = within
        # FIXME
        self.pattern = "{ObjectA} is on {ObjectB}." if not noprompt else "{ObjectA} {ObjectB}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            textA = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]
            
            # randomly choose an object
            # if within is True objectA is same as objectB else they are different
            arglabel = np.argwhere(labels.reshape(-1,)==label).reshape(-1,) if self.within else np.arange(labels.shape[0])

            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0]]
            labelB = labels[arglabel[0]][0]
            textB = sample["all_class_names"][labelB]

            # randomly shift first object upward
            pointsA = yshift(pointsA, ratio=BETA)

            # combine objects
            points = np.vstack((pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = ycentralizer(points)
            sample["points"] = points
            sample["text"] = self.pattern.format(ObjectA=textA, ObjectB=textB)
        return sample


class AObjectAIsUnderObjectB(object):
    """ A ObjectA is under to ObjectB.
    """
    def __init__(self, p: float = 0.5, within: bool = False, noprompt: bool = False) -> None:
        self.p = p
        self.within = within
        # FIXME
        self.pattern = "{ObjectA} is under {ObjectB}." if not noprompt else "{ObjectA} {ObjectB}"

    def __call__(self, sample: dict) -> dict:
        if self.p > random.random():
            # get another same object
            pointsA = sample["points"]
            textA = sample["text"]
            label = sample["label"]
            data = sample["data"]
            labels = sample["labels"]
            
            # randomly choose an object
            # if within is True objectA is same as objectB else they are different
            # arglabel = np.arange(labels.shape[0])
            arglabel = np.argwhere(labels.reshape(-1,)==label).reshape(-1,) if self.within else np.arange(labels.shape[0])

            np.random.shuffle(arglabel)
            pointsB = data[arglabel[0]]
            labelB = labels[arglabel[0]][0]
            textB = sample["all_class_names"][labelB]

            # randomly shift first object upward
            pointsB = yshift(pointsB, ratio=BETA)

            # combine objects
            points = np.vstack((pointsA, pointsB))
            points = reduce_points(points, num_points=pointsA.shape[0])
            points = ycentralizer(points)
            # rnd = random.randint(0, 100)
            # visualization.export_color_point_cloud(points=points, obj_path=f"{rnd}11111111111.obj", color=[0.0, 0.0, 0.9])
            sample["points"] = points
            sample["text"] = self.pattern.format(ObjectA=textA, ObjectB=textB)
        return sample














def xshift(points: np.ndarray, ratio: float = 1.0, shift: float = 0.0, is_random: bool = False) -> np.ndarray:
    """
    shift points in x direction
    ratio: shift proportional to width of object in x axis
    shift: absulute value to shift
    is_random: if is_random is true -> shift proportional to width of object in x axis by a random factor between [1.5,3.5)
    """
    if is_random:
        rndshft = 1.5 + 2 * random.random()
        xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
        shift = xspan * rndshft
    else:
        if shift == 0.0:
            xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
            shift = ratio * xspan

    if points.shape[1] == 6:
        points += np.array([shift, 0.0, 0.0, shift, 0.0, 0.0])
    else:
        points += np.array([shift, 0.0, 0.0])

    return points


def yshift(points: np.ndarray, ratio: float = 1.0, shift: float = 0.0, is_random: bool = False) -> np.ndarray:
    """
    shift points in y direction
    ratio: shift proportional to width of object in y axis
    shift: absulute value to shift
    is_random: if is_random is true -> shift proportional to width of object in y axis by a random factor between [1.5,3.5)
    """
    if is_random:
        rndshft = 1.5 + 2 * random.random()
        xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
        shift = xspan * rndshft
    else:
        if shift == 0.0:
            xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
            shift = ratio * xspan

    if points.shape[1] == 6:
        points += np.array([0.0, shift, 0.0, 0.0, shift, 0.0])
    else:
        points += np.array([0.0, shift, 0.0])
    
    return points

def zshift(points: np.ndarray, ratio: float = 1.0, shift: float = 0.0, is_random: bool = False) -> np.ndarray:
    """
    shift points in z direction
    ratio: shift proportional to width of object in z axis
    shift: absulute value to shift
    is_random: if is_random is true -> shift proportional to width of object in z axis by a random factor between [1.5,3.5)
    """
    if is_random:
        rndshft = 1.5 + 2 * random.random()
        xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
        shift = xspan * rndshft
    else:
        if shift == 0.0:
            xspan = points.max(axis=0)[0] - points.min(axis=0)[0]
            shift = ratio * xspan

    if points.shape[1] == 6:
        points += np.array([0.0, 0.0, shift, 0.0, 0.0, shift])
    else:
        points += np.array([0.0, 0.0, shift])

    return points


def reduce_points(points: np.ndarray, num_points: int = 1024) -> np.ndarray:
    """
    Randomly select num_points
    """
    idx = np.arange(points.shape[0])
    np.random.shuffle(idx)
    return points[idx[:num_points],:]


def xzcentralizer(points: np.ndarray) -> np.ndarray:
    """
    Zero mean points in xz direction
    """
    centroid = np.mean(points, axis=0)
    # no changes in y direction
    centroid[1] = 0
    return points - centroid

def ycentralizer(points: np.ndarray) -> np.ndarray:
    """
    Zero mean points in y direction
    """
    centroid = np.mean(points, axis=0)
    # no changes in y direction
    centroid[0] = 0
    centroid[2] = 0
    return points - centroid
