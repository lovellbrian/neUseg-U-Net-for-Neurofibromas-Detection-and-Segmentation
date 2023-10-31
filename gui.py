import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageOps, ImageQt
from unet import Unet


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(833, 749)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(400, 400))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(400, 400))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_4.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # store the number of skin and NF1 pixels
        self.skinnum = 0 
        self.NF1num = 0 

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.open_img)
        self.pushButton_2.clicked.connect(self.seg_skin)
        self.pushButton_3.clicked.connect(self.seg_NF1)
        self.pushButton_4.clicked.connect(self.ratio)

    # select image file and display image in GUI
    def open_img(self):
        self.img_file, _ = QFileDialog.getOpenFileName(self.centralwidget, 'Open file',
                                                       r'\\',
                                                       'Image files (*.jpg *.jpeg *.JPG)')

        self.textEdit.append('img_path：'+self.img_file)
        self.img=Image.open(self.img_file)
        self.img = ImageOps.exif_transpose(self.img)
        self.qimage = ImageQt.toqimage(self.img)

        qpixmap = QPixmap.fromImage(self.qimage)
        self.label_2.setPixmap(qpixmap)
        self.label_2.setScaledContents(True)

    # segment skin and display segmented image
    def seg_skin(self):
        unet = Unet()
        r_image,nums= unet.detect_image(self.img, model_path='model_data/skin.h5', count=True,
                                    name_classes=["background", "skin"])
        print(r_image)
        self.skinnum = nums[1]
        self.textEdit.append("[background, skin]: "+str(nums))
        self.r_image = ImageQt.toqimage(r_image)

        r_image = QPixmap.fromImage(self.r_image)
        self.label_3.setPixmap(r_image)
        self.label_3.setScaledContents(True)

    # segment NF1 aand display segmented image
    def seg_NF1(self):
        unet = Unet()
        r_image,nums= unet.detect_image(self.img, model_path='model_data/NF1.h5', count=True,
                                    name_classes=["background", "NF1"])
        print(r_image)
        self.NF1num = nums[1]
        self.textEdit.append("[background, cNF]: "+str(nums))
        self.r_image = ImageQt.toqimage(r_image)

        r_image = QPixmap.fromImage(self.r_image)
        self.label_3.setPixmap(r_image)
        self.label_3.setScaledContents(True)

    # calculate ratio of cNF pixels to skin pixels (estimate severity)
    def ratio(self):
        self.ratio=self.NF1num/self.skinnum
        self.textEdit.append('cNF-to-skin ratio: '+str(self.ratio))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "neUseg"))
        self.label.setText(_translate("MainWindow", "Neurofibroma Segmentation Software"))
        self.label_2.setText(_translate("MainWindow", "Original Image"))
        self.label_3.setText(_translate("MainWindow", "Segmented Image"))
        self.pushButton.setText(_translate("MainWindow", "Input Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Skin segmentation"))
        self.pushButton_3.setText(_translate("MainWindow", "cNF segmentation"))
        self.pushButton_4.setText(_translate("MainWindow", "cNF-to-skin ratio"))


if __name__ == '__main__':
    import PyQt5
    app = QApplication(sys.argv)
    ex = Ui_MainWindow()
    window = PyQt5.QtWidgets.QMainWindow()
    ex.setupUi(window)
    window.show()
    sys.exit(app.exec_())