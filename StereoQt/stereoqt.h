#ifndef STEREOQT_H
#define STEREOQT_H

#include <QtWidgets/QMainWindow>
#include "ui_stereoqt.h"

class StereoQt : public QMainWindow
{
	Q_OBJECT

public:
	StereoQt(QWidget *parent = 0);
	~StereoQt();

private:
	Ui::StereoQtClass ui;
};

#endif // STEREOQT_H
