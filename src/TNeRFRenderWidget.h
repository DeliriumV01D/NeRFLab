#ifndef TNERF_RENDER_WIDGET_H
#define TNERF_RENDER_WIDGET_H

#include "TThreadedNeRFExecutor.h"

#include <QtWidgets/QWidget>
#include <QThread.h>
#include <QtCore/QMutex.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


struct TRenderWidgetViewParams 
{
	bool DrawNeRFRGB{true},
		DrawNeRFDepth{false},
		DrawNeRFDisp{false};
};

class TNeRFRenderWidget : public QWidget {
	Q_OBJECT
private:

protected:
	QMutex NRWMutex;

	TThreadedNeRFExecutor	Executor;
	//std::unique_ptr<TThreadedNeRFExecutor> Executor = nullptr;
	torch::Tensor K;
	RenderParams RParams;
	TRenderWidgetViewParams ViewParams;
	cv::Mat RenderMat;

	float XRot,
		YRot,
		ZRot,
		XTra,
		YTra,
		ZTra,
		NSca;
	QPoint MousePosition;
	Qt::MouseButton MouseButtonPressed;

	void mousePressEvent(QMouseEvent * evnt)	override;
	void mouseMoveEvent(QMouseEvent * evnt) override;
	void mouseReleaseEvent(QMouseEvent * evnt) override;
	void wheelEvent(QWheelEvent * evnt) override;
	void keyPressEvent(QKeyEvent * evnt) override;

	void paintEvent(QPaintEvent *)	override;
	//void	repaint(int x, int y, int w, int h)
	//void	repaint(const QRect &rect)
	//void	repaint(const QRegion &rgn)
	//void	resize(const QSize &)
	//void	resize(int w, int h)
public:
	float SceneBoundingSphereRadius = 1;

	TNeRFRenderWidget(QWidget * parent = nullptr);
	~TNeRFRenderWidget();

	void SetExecutor(std::unique_ptr<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall>> &executor);
	void SetK(torch::Tensor k){K = k;};
	void SetDefaultPose(torch::Tensor default_pose);
	void SetRenderParams(const RenderParams &params);
	TRenderWidgetViewParams GetViewParams();
	void SetViewParams(const TRenderWidgetViewParams &params);
	torch::Tensor GetRenderPose();
	void SetRenderMat(const cv::Mat render_mat);
	void Render();
	
	void ScalePlus();
	void ScaleMinus();
	void RotateUp();
	void RotateDown();
	void RotateLeft();
	void RotateRight();
	void TranslateDown();
	void TranslateUp();
	void SetDefaultScene();
public slots:
	void OnUpdateResult(RenderResult render_result);
};

#endif