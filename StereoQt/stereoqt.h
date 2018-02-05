#ifndef STEREOQT_H
#define STEREOQT_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/qbuttongroup.h>
#include "ui_stereoqt.h"
#include "match.h"

class StereoQt : public QMainWindow
{
	Q_OBJECT

public:
	StereoQt(QWidget *parent = 0);
	~StereoQt();

private:
	Ui::StereoQtClass ui;
	QButtonGroup *methodsGroup;


	Match match;

	cv::Mat left;
	cv::Mat right;

	QString method_str;


	void LoadLeftImage();
	void LoadRightImage();
	void SetMehod();
	void Compute();
	void SaveResult();
	void ChangeLocalParams(int);

	void SetMatchParams();
};

#endif // STEREOQT_H
