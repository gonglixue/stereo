#include "stereoqt.h"
#include <QtWidgets/qfiledialog.h>

StereoQt::StereoQt(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	methodsGroup = new QButtonGroup(this);
	methodsGroup->addButton(ui.sadRadio, 0);
	methodsGroup->addButton(ui.nccRadio, 1);
	methodsGroup->addButton(ui.gcRadio, 2);
	ui.gcRadio->setChecked(true);

	connect(ui.loadLeftBtn, &QPushButton::clicked, this, &StereoQt::LoadLeftImage);
	connect(ui.loadRightBtn, &QPushButton::clicked, this, &StereoQt::LoadRightImage);

}

StereoQt::~StereoQt()
{

}

void StereoQt::LoadLeftImage()
{
	QString file_fn = QFileDialog::getOpenFileName(
		this,
		tr("Open An Image"),
		QString(),
		tr("Image Files(*.bmp, *.jpg, *.png)")
	);

	if (!file_fn.isEmpty())
	{
		this->left = cv::imread(file_fn.toStdString());
		cv::cvtColor(left, left, CV_BGR2RGB);

		QImage temp_qimage((const uchar*)left.data, left.cols, left.rows, left.step, QImage::Format_RGB888);
		this->ui.leftPosBtn->setIcon(QPixmap::fromImage(temp_qimage));
		this->ui.leftPosBtn->setIconSize(this->ui.leftPosBtn->size());
	}
}

void StereoQt::LoadRightImage()
{
	QString file_fn = QFileDialog::getOpenFileName(
		this,
		tr("Open An Image"),
		QString(),
		tr("Image Files(*.bmp, *.jpg, *.png)")
	);

	if (!file_fn.isEmpty())
	{
		this->right = cv::imread(file_fn.toStdString());
		cv::cvtColor(right, right, CV_BGR2RGB);

		QImage temp_qimage((const uchar*)right.data, right.cols, right.rows, right.step, QImage::Format_RGB888);
		this->ui.rightPosBtn->setIcon(QPixmap::fromImage(temp_qimage));
		this->ui.rightPosBtn->setIconSize(this->ui.rightPosBtn->size());
	}
}
