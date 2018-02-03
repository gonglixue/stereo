/********************************************************************************
** Form generated from reading UI file 'testqt.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TESTQT_H
#define UI_TESTQT_H

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

class Ui_testqtClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *testqtClass)
    {
        if (testqtClass->objectName().isEmpty())
            testqtClass->setObjectName(QStringLiteral("testqtClass"));
        testqtClass->resize(600, 400);
        menuBar = new QMenuBar(testqtClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        testqtClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(testqtClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        testqtClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(testqtClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        testqtClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(testqtClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        testqtClass->setStatusBar(statusBar);

        retranslateUi(testqtClass);

        QMetaObject::connectSlotsByName(testqtClass);
    } // setupUi

    void retranslateUi(QMainWindow *testqtClass)
    {
        testqtClass->setWindowTitle(QApplication::translate("testqtClass", "testqt", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class testqtClass: public Ui_testqtClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TESTQT_H
