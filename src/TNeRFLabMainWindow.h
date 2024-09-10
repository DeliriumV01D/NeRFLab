#ifndef TNERFLABMAINWINDOW_H
#define TNERFLABMAINWINDOW_H

#include <QMainWindow>
#include <QCheckBox>

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
		* acNeRFDisp;

public:
	TNeRFLabMainWindow(QWidget *parent = nullptr);
	~TNeRFLabMainWindow();

public slots:
	void OnActionOpenTriggered();
	void OnActionSaveTriggered();
	void OnActionSaveAsTriggered();

	void OnAcNeRFRGBTriggered(bool checked);
	void OnAcNeRFDepthTriggered(bool checked);
	void OnAcNeRFDispTriggered(bool checked);

};
#endif // TNERFLABMAINWINDOW_H
