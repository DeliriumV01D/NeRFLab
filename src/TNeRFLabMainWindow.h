#ifndef TNERFLABMAINWINDOW_H
#define TNERFLABMAINWINDOW_H

#include "load_blender.h"

#include <QMainWindow>
#include <QCheckBox>
#include <QTextEdit>

namespace tt{
class TabToolbar;
class Page;
}

QT_BEGIN_NAMESPACE
namespace Ui { class TNeRFLabMainWindow; }
QT_END_NAMESPACE

class TNeRFLabMainWindow : public QMainWindow
{
	Q_OBJECT
private:
	Ui::TNeRFLabMainWindow * ui;
	tt::TabToolbar * NeRFLabMainWindowToolbar;
	QAction * acNeRFRGB,
		* acNeRFDepth,
		* acNeRFDisp,
		* acLeRF;
	std::string DatasetDir,
		NeRFDir;
	NeRFDatasetParams Data;
	QTextEdit * tePrompt;

public:
	TNeRFLabMainWindow(QWidget *parent = nullptr);
	~TNeRFLabMainWindow();

public slots:
	void OnAcOpenImageFolderTriggered();
	void OnAcOpenBlenderDatasetTriggered();
	void OnAcOpenColmapReconstructionTriggered();
	void OnActionSaveTriggered();
	void OnActionSaveAsTriggered();

	void OnActionOpenNerfTriggered();
	void OnActionSaveNerfTriggered();
	void OnActionTrainNerfTriggered();
	void OnActionTrainLerfTriggered();

	void OnAcNeRFRGBTriggered(bool checked);
	void OnAcNeRFDepthTriggered(bool checked);
	void OnAcNeRFDispTriggered(bool checked);
	void OnAcLeRFTriggered(bool checked);

	void OnTePromptTextChanged();
};
#endif // TNERFLABMAINWINDOW_H
