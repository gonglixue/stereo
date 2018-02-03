#ifndef TESTQT_H
#define TESTQT_H

#include <QtWidgets/QMainWindow>
#include "ui_testqt.h"

class testqt : public QMainWindow
{
	Q_OBJECT

public:
	testqt(QWidget *parent = 0);
	~testqt();

private:
	Ui::testqtClass ui;
};

#endif // TESTQT_H
