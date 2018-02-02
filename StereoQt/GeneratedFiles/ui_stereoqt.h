/********************************************************************************
** Form generated from reading UI file 'stereoqt.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_STEREOQT_H
#define UI_STEREOQT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE
#define QStringLiteral(a) a

class Ui_StereoQtClass
{
public:
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QPushButton *loadLeftBtn;
    QPushButton *loadRightBtn;
    QRadioButton *sadRadio;
    QRadioButton *nccRadio;
    QRadioButton *gcRadio;
    QPushButton *computeBtn;
    QPushButton *saveBtn;
    QLabel *methodLabel;
    QTabWidget *tabWidget;
    QWidget *sadTab;
    QWidget *nccTab;
    QWidget *gcTab;
    QSpinBox *dispMinSpinBox;
    QSpinBox *dispMaxSpinBox;
    QSpinBox *iterSpinBox;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QWidget *imgWidget;
    QPushButton *leftPosBtn;
    QPushButton *rightPosBtn;
    QPushButton *dispPosBtn;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *StereoQtClass)
    {
        if (StereoQtClass->objectName().isEmpty())
            StereoQtClass->setObjectName(QStringLiteral("StereoQtClass"));
        StereoQtClass->resize(988, 500);
        centralWidget = new QWidget(StereoQtClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(20, 20, 221, 411));
        loadLeftBtn = new QPushButton(groupBox);
        loadLeftBtn->setObjectName(QStringLiteral("loadLeftBtn"));
        loadLeftBtn->setGeometry(QRect(20, 20, 151, 28));
        loadRightBtn = new QPushButton(groupBox);
        loadRightBtn->setObjectName(QStringLiteral("loadRightBtn"));
        loadRightBtn->setGeometry(QRect(20, 60, 151, 28));
        sadRadio = new QRadioButton(groupBox);
        sadRadio->setObjectName(QStringLiteral("sadRadio"));
        sadRadio->setGeometry(QRect(40, 122, 51, 19));
        nccRadio = new QRadioButton(groupBox);
        nccRadio->setObjectName(QStringLiteral("nccRadio"));
        nccRadio->setGeometry(QRect(40, 148, 51, 19));
        gcRadio = new QRadioButton(groupBox);
        gcRadio->setObjectName(QStringLiteral("gcRadio"));
        gcRadio->setGeometry(QRect(40, 174, 99, 19));
        computeBtn = new QPushButton(groupBox);
        computeBtn->setObjectName(QStringLiteral("computeBtn"));
        computeBtn->setGeometry(QRect(20, 340, 151, 28));
        saveBtn = new QPushButton(groupBox);
        saveBtn->setObjectName(QStringLiteral("saveBtn"));
        saveBtn->setGeometry(QRect(20, 380, 151, 28));
        methodLabel = new QLabel(groupBox);
        methodLabel->setObjectName(QStringLiteral("methodLabel"));
        methodLabel->setGeometry(QRect(21, 100, 171, 16));
        tabWidget = new QTabWidget(groupBox);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(0, 200, 221, 131));
        sadTab = new QWidget();
        sadTab->setObjectName(QStringLiteral("sadTab"));
        tabWidget->addTab(sadTab, QString());
        nccTab = new QWidget();
        nccTab->setObjectName(QStringLiteral("nccTab"));
        tabWidget->addTab(nccTab, QString());
        gcTab = new QWidget();
        gcTab->setObjectName(QStringLiteral("gcTab"));
        dispMinSpinBox = new QSpinBox(gcTab);
        dispMinSpinBox->setObjectName(QStringLiteral("dispMinSpinBox"));
        dispMinSpinBox->setGeometry(QRect(90, 10, 46, 22));
        dispMinSpinBox->setMinimum(-80);
        dispMinSpinBox->setMaximum(0);
        dispMinSpinBox->setValue(-60);
        dispMaxSpinBox = new QSpinBox(gcTab);
        dispMaxSpinBox->setObjectName(QStringLiteral("dispMaxSpinBox"));
        dispMaxSpinBox->setGeometry(QRect(90, 40, 46, 22));
        iterSpinBox = new QSpinBox(gcTab);
        iterSpinBox->setObjectName(QStringLiteral("iterSpinBox"));
        iterSpinBox->setGeometry(QRect(90, 70, 46, 22));
        iterSpinBox->setMinimum(3);
        iterSpinBox->setMaximum(10);
        iterSpinBox->setValue(4);
        label = new QLabel(gcTab);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 10, 72, 15));
        label_2 = new QLabel(gcTab);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 40, 72, 15));
        label_3 = new QLabel(gcTab);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(0, 70, 81, 16));
        tabWidget->addTab(gcTab, QString());
        imgWidget = new QWidget(centralWidget);
        imgWidget->setObjectName(QStringLiteral("imgWidget"));
        imgWidget->setGeometry(QRect(270, 20, 691, 411));
        leftPosBtn = new QPushButton(imgWidget);
        leftPosBtn->setObjectName(QStringLiteral("leftPosBtn"));
        leftPosBtn->setEnabled(false);
        leftPosBtn->setGeometry(QRect(20, 20, 200, 150));
        rightPosBtn = new QPushButton(imgWidget);
        rightPosBtn->setObjectName(QStringLiteral("rightPosBtn"));
        rightPosBtn->setEnabled(false);
        rightPosBtn->setGeometry(QRect(20, 210, 200, 150));
        dispPosBtn = new QPushButton(imgWidget);
        dispPosBtn->setObjectName(QStringLiteral("dispPosBtn"));
        dispPosBtn->setEnabled(false);
        dispPosBtn->setGeometry(QRect(230, 20, 441, 341));
        StereoQtClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(StereoQtClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 988, 26));
        StereoQtClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(StereoQtClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        StereoQtClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(StereoQtClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        StereoQtClass->setStatusBar(statusBar);

        retranslateUi(StereoQtClass);

        tabWidget->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(StereoQtClass);
    } // setupUi

    void retranslateUi(QMainWindow *StereoQtClass)
    {
        StereoQtClass->setWindowTitle(QApplication::translate("StereoQtClass", "StereoQt", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("StereoQtClass", "Control", Q_NULLPTR));
        loadLeftBtn->setText(QApplication::translate("StereoQtClass", "Load Left Image", Q_NULLPTR));
        loadRightBtn->setText(QApplication::translate("StereoQtClass", "Load Right Image", Q_NULLPTR));
        sadRadio->setText(QApplication::translate("StereoQtClass", "SAD", Q_NULLPTR));
        nccRadio->setText(QApplication::translate("StereoQtClass", "NCC", Q_NULLPTR));
        gcRadio->setText(QApplication::translate("StereoQtClass", "Graph Cut", Q_NULLPTR));
        computeBtn->setText(QApplication::translate("StereoQtClass", "Compute Disparity", Q_NULLPTR));
        saveBtn->setText(QApplication::translate("StereoQtClass", "Save Disparity", Q_NULLPTR));
        methodLabel->setText(QApplication::translate("StereoQtClass", "Choose An Algorithm:", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        sadTab->setToolTip(QApplication::translate("StereoQtClass", "<html><head/><body><p>SAD \345\217\202\346\225\260</p></body></html>", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        tabWidget->setTabText(tabWidget->indexOf(sadTab), QApplication::translate("StereoQtClass", "1", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(nccTab), QApplication::translate("StereoQtClass", "2", Q_NULLPTR));
        label->setText(QApplication::translate("StereoQtClass", "dispMin", Q_NULLPTR));
        label_2->setText(QApplication::translate("StereoQtClass", "dispMax", Q_NULLPTR));
        label_3->setText(QApplication::translate("StereoQtClass", "Iter Times", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(gcTab), QApplication::translate("StereoQtClass", "3", Q_NULLPTR));
        leftPosBtn->setText(QString());
        rightPosBtn->setText(QString());
        dispPosBtn->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class StereoQtClass: public Ui_StereoQtClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STEREOQT_H
