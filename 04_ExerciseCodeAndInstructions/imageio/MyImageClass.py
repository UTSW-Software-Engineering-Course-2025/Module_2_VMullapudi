from re import L
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections.abc import Sequence

#define an image class that has useful functions we can use!

class MyImgClass():
    """
    This is the custom class for our lesson 2.
    It contains all the methods to run the accompanying notebook.
    Many you will be implementing.

    """
    
    def __init__(self, arrImg: np.ndarray, intLabel=None):
        """Initalize the MyImgClass object

        :param arrImg: a numpy array that contains the image
        :type arrImg: np.array
        :param intLabel: the label value, defaults to None
        :type intLabel: int, optional
        """
        self.arrImg = arrImg
        self.intLabel = intLabel
        self.shape = self.arrImg.shape
        
    def __add__(self, other: "MyImgClass") -> "MyImgClass":
        """
        operator overloading for the '+' operation 
        element wise addtion of two MyImgClass together.

        :param other: the other instance of MyImgClass
        :type other: MyImgClass
        :return: the addition of the two classes
        :rtype: MyImgClass
        """
        #toReturn = MyImgClass(np.add(self.arrImg, other.arrImg), intLabel=None)
        return MyImgClass(np.add(self.arrImg, other.arrImg), intLabel=None)
    
    def __sub__(self, other: "MyImgClass") -> "MyImgClass":
        """
        operator overloading for the '-' operation 
        element wise subtraction of two MyImgClass together.

        :param other: the other instance of MyImgClass
        :type other: MyImgClass
        :return: the subtraction of the two classes
        :rtype: MyImgClass
        """
        return MyImgClass(np.subtract(self.arrImg, other.arrImg), intLabel=None)
        
    
    def fPixelwiseSqDif(self, other: "MyImgClass") -> "MyImgClass":
        """Find the square difference between two MyImgClass objects

        :param other: the other instance of MyImgClass
        :type other: MyImgClass
        :return: square difference of each pixel
        :rtype: MyImgClass
        """
        
        return MyImgClass((self.arrImg - other.arrImg) ** 2)
    
    
    
    def fMSE(self, other: "MyImgClass") -> np.floating:
        """Find the mean squared error between two images

        :param other: the other instance of MyImgClass
        :type other: MyImgClass
        :return: MSE
        :rtype: float
        """
        return np.mean(self.fPixelwiseSqDif(other).arrImg)

    
    def fPlot(self, ax: plt.Axes, show_ticks=False, add_colorbar=False, imshow_kwargs={}):
        """Plotting method for the class

        :param ax: the axis to plot on
        :type ax: matplotlib.pyplot.ax
        :param show_ticks: display the x and y axis ticks/labels, defaults to False
        :type show_ticks: bool, optional
        :param add_colorbar: display a color bar, defaults to False
        :type add_colorbar: bool, optional
        :param imshow_kwargs: additional keywords, defaults to {}
        :type imshow_kwargs: dict, optional
        """
        img = ax.imshow(self.arrImg, interpolation='None', **imshow_kwargs)
        if not self.intLabel is None:
            ax.set_title(f'Label: {self.intLabel}')
        if show_ticks is False:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(img, cax=cax, orientation='vertical')        
        
    @staticmethod
    def fComputeMeanAcrossImages(lMyImgClass: Sequence["MyImgClass"]) -> "MyImgClass":
        """Calculate the mean image from a list of images

        :param lMyImgClass: list of MyImgClass
        :type lMyImgClass: Sequence["MyImgClass"]
        :return: The mean image
        :rtype: MyImgClass
        """
        if len(lMyImgClass) == 0:
            raise ValueError('Empty list provided. Please provide a list with at least one image.')
        if not all(isinstance(img, MyImgClass) for img in lMyImgClass):
            raise TypeError('All elements in the list must be instances of MyImgClass')
        
        # Stack images along a new dimension and compute the mean
        # Ensure all images have the same shape
        shapes = [img.arrImg.shape for img in lMyImgClass]
        if len(set(shapes)) > 1:
            raise ValueError('All images must have the same shape to compute the mean.')
        
        stacked_images = np.stack([img.arrImg for img in lMyImgClass], axis=0)
        return MyImgClass(np.mean(stacked_images, axis=0))
  
    
    @staticmethod
    def fComputeStdAcrossImages(lMyImgClass: Sequence["MyImgClass"]) -> "MyImgClass":
        """Calculate the std image from a list of images

        :param lMyImgClass: list of MyImgClass
        :type lMyImgClass: list (or 1D iterable)
        :return: The std image
        :rtype: MyImgClass
        """
        if len(lMyImgClass) == 0:
            raise ValueError('Empty list provided. Please provide a list with at least one image.')
        if not all(isinstance(img, MyImgClass) for img in lMyImgClass):
            raise TypeError('All elements in the list must be instances of MyImgClass')
        
        # Stack images along a new dimension and compute the mean
        # Ensure all images have the same shape
        shapes = [img.arrImg.shape for img in lMyImgClass]
        if len(set(shapes)) > 1:
            raise ValueError('All images must have the same shape to compute the mean.')
        
        stacked_images = np.stack([img.arrImg for img in lMyImgClass], axis=0)
        return MyImgClass(np.std(stacked_images, axis=0))
        

    @staticmethod
    def fMeanMSE(lImg1: Sequence["MyImgClass"], lImg2: Sequence["MyImgClass"]) -> np.floating:
        """Calcualte the mean MSE across pairs of images
        e.g. mean(MSE(lImg1[0],lImg2[0]),MSE(lImg1[1],lImg2[1])...)

        :param lImg1: list of MyImgClass
        :type lImg1: list (or 1D iterable)
        :param lImg2: list of MyImgClass
        :type lImg2: list (or 1D iterable)
        :return: mean MSE
        :rtype: float
        """
        return np.mean([img1.fMSE(img2) for img1, img2 in zip(lImg1, lImg2)])


    @staticmethod
    def fMSEforEachPairCombination(lImg1, lImg2):
        """Calcaulte the MSE for all pairs between lImg1 and lImg2

        :param lImg1: list of MyImgClass
        :type lImg1: list (or 1D iterable)
        :param lImg2: list of MyImgClass
        :type lImg2: list (or 1D iterable)
        :return: mean MSE
        :rtype: float
        """
        lSE = []
        for imgI in lImg1:
            for imgJ in lImg2:
                if imgI != imgJ:
                    lSE.append(imgI.fMSE(imgJ))
                else:
                    lSE.append(np.nan)
        return lSE

