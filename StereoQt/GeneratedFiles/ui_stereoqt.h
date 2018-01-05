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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_StereoQtClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *StereoQtClass)
    {
        if (StereoQtClass->objectName().isEmpty())
            StereoQtClass->setObjectName("StereoQtClass");
        StereoQtClass->resize(600, 400);
        menuBar = new QMenuBar(StereoQtClass);
        menuBar->setObjectName("menuBar");
        StereoQtClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(StereoQtClass);
        mainToolBar->setObjectName("mainToolBar");
        StereoQtClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(StereoQtClass);
        centralWidget->setObjectName("centralWidget");
        StereoQtClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(StereoQtClass);
        statusBar->setObjectName("statusBar");
        StereoQtClass->setStatusBar(statusBar);

        retranslateUi(StereoQtClass);

        QMetaObject::connectSlotsByName(StereoQtClass);
    } // setupUi

    void retranslateUi(QMainWindow *StereoQtClass)
    {
        StereoQtClass->setWindowTitle(QApplication::translate("StereoQtClass", "StereoQt", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class StereoQtClass: public Ui_StereoQtClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STEREOQT_H
