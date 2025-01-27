#include "stereoqt.h"
#include <QtWidgets/qfiledialog.h>
#include <qdebug.h>
#include <QtWidgets/qmessagebox.h>

static const int MAX_DENOM = 1 << 4;
/// Store in \a params fractions approximating the last 3 parameters.
///
/// They have the same denominator (up to \c MAX_DENOM), chosen so that the sum
/// of relative errors is minimized.
void set_fractions(Match::Parameters& params,
	float K, float lambda1, float lambda2) {
	float minError = std::numeric_limits<float>::max();
	for (int i = 1; i <= MAX_DENOM; i++) {
		float e = 0;
		int numK = 0, num1 = 0, num2 = 0;
		if (K>0)
			e += std::abs((numK = int(i*K + .5f)) / (i*K) - 1.0f);
		if (lambda1>0)
			e += std::abs((num1 = int(i*lambda1 + .5f)) / (i*lambda1) - 1.0f);
		if (lambda2>0)
			e += std::abs((num2 = int(i*lambda2 + .5f)) / (i*lambda2) - 1.0f);
		if (e<minError) {
			minError = e;
			params.denominator = i;
			params.K = numK;
			params.lambda1 = num1;
			params.lambda2 = num2;
		}
	}
}

/// Make sure parameters K, lambda1 and lambda2 are non-negative.
///
/// - K may be computed automatically and lambda set to K/5.
/// - lambda1=3*lambda, lambda2=lambda
/// As the graph requires integer weights, use fractions and common denominator.
void fix_parameters(Match& m, Match::Parameters& params,
	float& K, float& lambda, float& lambda1, float& lambda2) {
	if (K<0) { // Automatic computation of K
		m.SetParameters(&params);
		K = m.GetK();
	}
	if (lambda<0) // Set lambda to K/5
		lambda = K / 5;
	if (lambda1<0) lambda1 = 3 * lambda;
	if (lambda2<0) lambda2 = lambda;
	set_fractions(params, K, lambda1, lambda2);
	m.SetParameters(&params);
}

StereoQt::StereoQt(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	methodsGroup = new QButtonGroup(this);
	methodsGroup->addButton(ui.sadRadio, 0);
	methodsGroup->addButton(ui.nccRadio, 1);
	methodsGroup->addButton(ui.gcRadio, 2);
	method_str = "Graph Cuts";
	ui.gcRadio->setChecked(true);

	connect(ui.loadLeftBtn, &QPushButton::clicked, this, &StereoQt::LoadLeftImage);
	connect(ui.loadRightBtn, &QPushButton::clicked, this, &StereoQt::LoadRightImage);

	// methods radio
	connect(ui.sadRadio, &QRadioButton::clicked, this, &StereoQt::SetMehod);
	connect(ui.nccRadio, &QRadioButton::clicked, this, &StereoQt::SetMehod);
	connect(ui.gcRadio, &QRadioButton::clicked, this, &StereoQt::SetMehod);

	// compute
	connect(ui.computeBtn, &QRadioButton::clicked, this, &StereoQt::Compute);

	// save
	connect(ui.saveBtn, &QRadioButton::clicked, this, &StereoQt::SaveResult);

	// change parameters
	connect(ui.dispMinSpinBox_NCC, SIGNAL(valueChanged(int)), this, SLOT(ChangeLocalParams(int)));
	connect(ui.dispMinSpinBox_SAD, SIGNAL(valueChanged(int)), this, SLOT(ChangeLocalParams(int)));
	connect(ui.winSpinBox_NCC, SIGNAL(valueChanged(int)), this, SLOT(ChangeLocalParams(int)));
	connect(ui.winSpinBox_SAD, SIGNAL(valueChanged(int)), this, SLOT(ChangeLocalParams(int)));


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

void StereoQt::SetMehod()
{
	this->match.SetMehod(methodsGroup->checkedId());
	qDebug() << "set method " << (methodsGroup->checkedId()) << "\n";

	//QString method_str;
	if (this->methodsGroup->checkedId() == 0)
		method_str = "SAD";
	else if (this->methodsGroup->checkedId() == 1)
		method_str = "NCC";
	else
		method_str = "Graph Cuts";
}

void StereoQt::Compute()
{
	if (left.cols == 0 || right.cols == 0) {
		QMessageBox::warning(this,
			tr("Stereo"),
			tr("The left image or the right image is empty.\n"
				"Please load images first."),
			QMessageBox::Ok);
		return;
	}

	match.InitMatch(left, right);
	SetMatchParams();

	// match.KZ2(ui.progressBar);
	// match.SaveXLeft("D:/test.png");

	cv::Mat result = match.PerformMatchAllMethods(ui.progressBar);
	qDebug() << "compute done!\n";

	QImage temp_qimage((const uchar*)result.data, result.cols, result.rows, result.step, QImage::Format_Grayscale8);
	this->ui.dispPosBtn->setIcon(QPixmap::fromImage(temp_qimage));
	this->ui.dispPosBtn->setIconSize(this->ui.dispPosBtn->size());

	char info_str[200];
	sprintf(info_str, "Computation Compete.\n" 
		"Image Size:[%d, %d]\n"
		"Method:%s\n"
		"Time Cost:%f ms\n", left.cols, left.rows, method_str.toStdString(), match.time_ms);

	QMessageBox::information(
		this, tr("Stereo"), 
		tr(info_str
		), 
		QMessageBox::Ok,
		QMessageBox::Ok);

}

void StereoQt::SetMatchParams()
{
	Match::Parameters params = {
		Match::Parameters::L2, 1,
		8, -1, -1,
		-1,
		4, false
	};

	float K = -1, lambda = -1, lambda1 = -1, lambda2 = -1;

	match.SetDispRange(-60, 0);
	time_t seed = time(NULL);
	srand((unsigned int)seed);
	fix_parameters(match, params, K, lambda, lambda1, lambda2);
	
	Match::LocalParameters local_params = {
		64, 5
	};
	match.SetLocalParameters(&local_params);
}

void StereoQt::SaveResult()
{
	cv::Mat result = match.AccessFinalOut();
	if (result.cols == 0) {
		QMessageBox::warning(this,
			tr("Stereo"),
			tr("Nothing to save.\n"
				"Please load images and compute disparity first."),
			QMessageBox::Ok);
		return;
	}

	QString file_fn = QFileDialog::getSaveFileName(
		this,
		tr("Save Disparity Image"),
		QString(),
		tr("Image Files(*.png)")
	);
	if (file_fn.isEmpty())
		return;

	cv::imwrite(file_fn.toStdString(), result);
}

void StereoQt::ChangeLocalParams(int a)
{
	Match::LocalParameters local_params;
	if (this->methodsGroup->checkedId() == 1) {
		local_params.max_disparity = -1 * ui.dispMinSpinBox_NCC->value();
		local_params.win_size = ui.winSpinBox_NCC->value();
		match.SetLocalParameters(&local_params);
	}
	else if (this->methodsGroup->checkedId() == 0) {
		local_params.max_disparity = -1 * ui.dispMinSpinBox_SAD->value();
		local_params.win_size = ui.winSpinBox_SAD->value();
		match.SetLocalParameters(&local_params);
	}

	char info[10];
	sprintf(info, "win:%d", local_params.win_size);
	QMessageBox::information(
		this, tr("Stereo"),
		tr(info),
		QMessageBox::Ok,
		QMessageBox::Ok);
}