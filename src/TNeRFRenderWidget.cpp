#include "TNeRFRenderWidget.h"

#include <qevent.h>
#include <QDebug.h>
#include <QPainter.h>
#include <QtCore/QCoreApplication.h>

#include "CommonDefinitions.h"
#include "load_blender.h"		//GetCalibrationMatrix

///Версия 4x4 (в однородных координатах). Переписать в пакетном виде. xyz == 1 (нормализовать)
static torch :: Tensor AxisAngle(torch::Tensor axis, torch::Tensor angle)
{
	//if (inner_product(axis, 0.) != 0 and angle != 0)
	//	return;

	torch::Tensor result = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(axis.device()/*torch::kCPU*/));
	auto x = axis.index({0}),
		y = axis.index({1}),
		z = axis.index({2}),
		c = torch::cos(angle),
		s = torch::sin(angle),
		C = 1.f - c;
		
	result.index_put_({0, 0}, x * x * C + c); 
	result.index_put_({0, 1}, x * y * C - z * s); 
	result.index_put_({0, 2}, x * z * C + y * s);
	result.index_put_({1, 0}, y * x * C + z * s); 
	result.index_put_({1, 1}, y * y * C + c); 
	result.index_put_({1, 2}, y * z * C - x * s);
	result.index_put_({2, 0}, z * x * C - y * s); 
	result.index_put_({2, 1}, z * y * C + x * s); 
	result.index_put_({2, 2}, z * z * C + c);
	result.index_put_({3, 3}, 1.f);

	return result/*.to(torch::kCUDA)*/;
}

TNeRFRenderWidget :: TNeRFRenderWidget(QWidget * parent /*= nullptr*/)
	: QWidget(parent)
{
	SetDefaultScene();
	setFocusPolicy(Qt::StrongFocus);

	connect(&Executor, SIGNAL(UpdateResult(std::tuple<NeRFRenderResult, LeRFRenderResult>)), this, SLOT(OnUpdateResult(std::tuple<NeRFRenderResult, LeRFRenderResult>)));
}

TNeRFRenderWidget :: ~TNeRFRenderWidget()
{
}

void TNeRFRenderWidget :: SetExecutor(std::unique_ptr<TThreadedNeRFExecutor::TExecutor> &executor, const bool render /*= true*/)
{
	torch::Tensor bounding_box;
	if (executor->GetParams().use_nerf)
		bounding_box = executor->GetEmbedderBoundingBox();
	if (executor->GetParams().use_lerf)
		bounding_box = executor->GetLangEmbedderBoundingBox();
	std::vector<torch::Tensor> splits = torch::split(bounding_box, { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];

	NRWMutex.lock();
	SceneBoundingSphereRadius = (box_max - box_min).norm().item<float>()/2;
	SetDefaultScene();
	NRWMutex.unlock();

	Executor.SetExecutor(executor);

	if (render) Render();
}


void TNeRFRenderWidget::SetDefaultPose(torch::Tensor default_pose, const bool render /*= true*/)
{
	//Вот это уже делается раньше при загрузки pose из Colmap
	////w2c->c2w
	//torch::Tensor R_inv = torch::linalg_inv(R_tens);	//R.transpose(0, 1); // Для ортогональной матрицы вращения обратная = транспонированная
	//torch::Tensor t_inv = -torch::matmul(R_inv, t_tens);
	////Convert from COLMAP's camera coordinate system (OpenCV) to NeRF (OpenGL) | righthanded <-> lefthanded
	//pose.index({ torch::indexing::Slice(0, 3), torch::indexing::Slice(1, 3) }) *= -1;

	//Перейдем из системы координат связанной с камерой в мировую систему координат
	torch::Tensor default_pose_world = C2W2C(default_pose);

	torch::Tensor R = default_pose_world.index({ torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3) });
	//Извлекаем углы Эйлера в порядке ZYX (соответствует порядку применения в GetRenderPose)
	float found_x_rot, found_y_rot, found_z_rot;
	float sy = -R.index({ 2, 0 }).item<float>();
	const float eps = 1e-6;
	if (std::fabs(sy) < 1.0f - eps)
	{
		found_x_rot = std::atan2(R.index({ 2, 1 }).item<float>(), R.index({ 2, 2 }).item<float>());
		found_y_rot = std::asin(sy);
		found_z_rot = std::atan2(R.index({ 1, 0 }).item<float>(), R.index({ 0, 0 }).item<float>());
	} else {
		//Обработка случая gimbal lock
		found_x_rot = std::atan2(-R.index({ 1, 2 }).item<float>(), R.index({ 1, 1 }).item<float>());
		found_y_rot = (sy > 0) ? PI / 2.0f : -PI / 2.0f;
		found_z_rot = 0.0f;
	}
	float tx = default_pose_world.index({ 0, 3 }).item<float>();
	float ty = default_pose_world.index({ 1, 3 }).item<float>();
	float tz = default_pose_world.index({ 2, 3 }).item<float>();

	NRWMutex.lock();
	XTra = tx;
	YTra = ty;
	ZTra = tz;
	XRot = found_x_rot * 180.0f / PI;
	YRot = found_y_rot * 180.0f / PI;
	ZRot = found_z_rot * 180.0f / PI;
	NSca = 1.0f;
	NRWMutex.unlock();

	if (render) Render();
}

void TNeRFRenderWidget :: SetRenderParams(const NeRFRenderParams &params, const bool render /*= true*/)
{
	NRWMutex.lock();
	RParams = params;
	std::vector<torch::Tensor> splits = torch::split(RParams.BoundingBox, { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];
	SceneBoundingSphereRadius = (box_max - box_min).norm().item<float>() / 2;
	//SetDefaultScene();
	NRWMutex.unlock();

	if (render) Render();
};

TRenderWidgetViewParams TNeRFRenderWidget :: GetViewParams()
{
	TRenderWidgetViewParams result;
	NRWMutex.lock();
	result = ViewParams;
	NRWMutex.unlock();
	return result;
};

void TNeRFRenderWidget :: SetViewParams(const TRenderWidgetViewParams &params, const bool render /*= true*/)
{
	NRWMutex.lock();
	ViewParams = params;
	NRWMutex.unlock();

	if (render) Render();
};

torch::Tensor TNeRFRenderWidget :: GetRenderPose()
{
	torch::NoGradGuard no_grad;

	NRWMutex.lock();
	torch::Tensor pose;
	try {
		//Загружаем единичную матрицу моделировани
		float pose_world_data[] = { 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1 };
		auto pose_world = torch::from_blob(pose_world_data, { 4, 4 });
		//Масштабирование
		float scale_data[] = { NSca, 0, 0, 0,
			0, NSca, 0, 0,
			0, 0, NSca, 0,
			0, 0, 0, 1};
		auto scale = torch::from_blob(scale_data, { 4, 4 });
		pose_world = torch::matmul(pose_world, scale);

		//Повороты
		pose_world = torch::matmul(pose_world, AxisAngle(torch::tensor({ 1.0f, 0.0f, 0.0f }), torch::tensor({ XRot / 180 * PI })));  // Pitch  
		pose_world = torch::matmul(pose_world, AxisAngle(torch::tensor({ 0.0f, 1.0f, 0.0f }), torch::tensor({ YRot / 180 * PI })));  // Yaw
		pose_world = torch::matmul(pose_world, AxisAngle(torch::tensor({ 0.0f, 0.0f, 1.0f }), torch::tensor({ ZRot / 180 * PI })));  // Roll
		//Трансляции
		float trans_data[] = { 1, 0, 0, XTra,
			0, 1, 0, YTra,
			0, 0, 1, ZTra,
			0, 0, 0, 1};
		auto trans = torch::from_blob(trans_data, { 4, 4 });
		//pose = torch::matmul(pose, trans);
		pose_world.index_put_({ torch::indexing::Slice(0, 3), 3 }, trans.index({ torch::indexing::Slice(0, 3), 3 }));
		//Перейдем из мировой системы координат в систему координат связанную с камерой
		pose = C2W2C(pose_world);
	} catch (std::exception &e) {
		NRWMutex.unlock();
		throw e;
	};
	NRWMutex.unlock();

	return pose;
}

void TNeRFRenderWidget :: SetRenderMat(const cv::Mat render_mat)
{
	NRWMutex.lock();
	render_mat.copyTo(RenderMat);
	NRWMutex.unlock();
}

void TNeRFRenderWidget :: Render()
{
	auto pose = GetRenderPose();
	
	int h = this->size().height(), 
		w = this->size().width();
	
	//torch::NoGradGuard no_grad;
	//RenderResult render_result = Executor.Executor->RenderView(pose, w, h, K, RenderParams);

	Executor.RenderView(pose, w, h, K, RParams);
}


void TNeRFRenderWidget :: mousePressEvent(QMouseEvent * evnt)
{
	MousePosition = evnt->pos();
	MouseButtonPressed = evnt->button();
}

void TNeRFRenderWidget::mouseMoveEvent(QMouseEvent* evnt)
{
	NRWMutex.lock();
	if (MouseButtonPressed == Qt::RightButton)
	{
		//Вертикальное движение - вращение вокруг X (Pitch)
		XRot += 180 * (float)(evnt->y() - MousePosition.y()) / height();
		//Горизонтальное движение - вращение вокруг Y (Yaw)  
		YRot -= 180 * (float)(evnt->x() - MousePosition.x()) / width();
		////Ограничиваем Pitch чтобы избежать переворота
		//XRot = std::clamp(XRot, -89.0f, 89.0f);
	}

	if (MouseButtonPressed == Qt::LeftButton)
	{
		XTra += (float)(evnt->x() - MousePosition.x()) / std::min(width(), height()) / NSca * SceneBoundingSphereRadius;
		YTra -= (float)(evnt->y() - MousePosition.y()) / std::min(width(), height()) / NSca * SceneBoundingSphereRadius;
	}
	NRWMutex.unlock();

	MousePosition = evnt->pos();

	Render();
}

void TNeRFRenderWidget :: mouseReleaseEvent(QMouseEvent * evnt)
{
}

void TNeRFRenderWidget :: wheelEvent(QWheelEvent * evnt)
{
	NRWMutex.lock();
	if ((evnt->angleDelta().y()) < 0) ScaleMinus();
	else if ((evnt->angleDelta().y()) > 0) ScalePlus();
	NRWMutex.unlock();

	Render();
}

void TNeRFRenderWidget :: keyPressEvent(QKeyEvent * evnt)
{
	NRWMutex.lock();
	switch (evnt->key())
	{
	case Qt::Key_Plus:
		ScalePlus();
		break;

	case Qt::Key_Equal:
		ScalePlus();
		break;

	case Qt::Key_Minus:
		ScaleMinus();
		break;

	case Qt::Key_Up:
		RotateUp();
		break;

	case Qt::Key_Down:
		RotateDown();
		break;

	case Qt::Key_Left:
		RotateLeft();
		break;

	case Qt::Key_Right:
		RotateRight();
		break;

	case Qt::Key_W:
		TranslateForward();
		break;
	case Qt::Key_S:
		TranslateBackward();
		break;
	case Qt::Key_D:
		TranslateRight();
		break;
	case Qt::Key_A:
		TranslateLeft();
		break;

	case Qt::Key_Z:
		TranslateDown();
		break;

	case Qt::Key_X:
		TranslateUp();
		break;

	case Qt::Key_Space:
		SetDefaultScene();
		break;

	case Qt::Key_Escape:
		this->close();
		break;
	}
	NRWMutex.unlock();

	Render();
}

void TNeRFRenderWidget :: paintEvent(QPaintEvent * evt)
{
	QPainter painter(this);
	QRect target(0, 0, this->size().width(), this->size().height());

	NRWMutex.lock();
	painter.drawPixmap(target, cvMatToQPixmap(RenderMat));
	NRWMutex.unlock();
}

void TNeRFRenderWidget :: resizeEvent(QResizeEvent* event)
{
	if (K.defined() && (K.numel() != 0))
	{
		SetK(GetSameFOVCalibrationMatrix(K, event->size().width(), event->size().height()));
		Render();
	}
}

void TNeRFRenderWidget :: ScalePlus()
{
	NSca = NSca * 1.1f;
}

void TNeRFRenderWidget :: ScaleMinus()
{
	NSca = NSca / 1.1f;
}

void TNeRFRenderWidget :: RotateUp()
{
	XRot += 1.0f;
}

void TNeRFRenderWidget :: RotateDown()
{
	XRot -= 1.0f;
}

void TNeRFRenderWidget :: RotateLeft()
{
	ZRot -= 1.0f;
}

void TNeRFRenderWidget :: RotateRight()
{
	ZRot += 1.0f;
}

void TNeRFRenderWidget::TranslateForward()
{
	ZTra += 0.05f;
}

void TNeRFRenderWidget::TranslateBackward()
{
	ZTra -= 0.05f;
}

void TNeRFRenderWidget::TranslateRight()
{
	XTra -= 0.05f;
}

void TNeRFRenderWidget :: TranslateLeft()
{
	XTra += 0.05f;
}

void TNeRFRenderWidget :: TranslateDown()
{
	ZTra -= 0.05f;
}

void TNeRFRenderWidget :: TranslateUp()
{
	ZTra += 0.05f;
}

void TNeRFRenderWidget :: SetDefaultScene()
{
	XTra = 0;
	YTra = 0;
	ZTra = 0;
	XRot = 0;
	YRot = 0;//180;
	ZRot = 180;
	NSca = 1;
}

//SLOTS

void TNeRFRenderWidget :: OnUpdateResult(std::tuple<NeRFRenderResult, LeRFRenderResult> render_result)
{
	if (ViewParams.DrawNeRFRGB)
		if (std::get<0>(render_result).Outputs1.RGBMap.defined())
		{
			SetRenderMat(TorchTensorToCVMat(std::get<0>(render_result).Outputs1.RGBMap.cpu()));
		}

	if (ViewParams.DrawNeRFDepth)
		if (std::get<0>(render_result).Outputs1.DepthMap.defined())
		{
			std::get<0>(render_result).Outputs1.DepthMap = (std::get<0>(render_result).Outputs1.DepthMap - RParams.Near) / (RParams.Far - RParams.Near);
			SetRenderMat(TorchTensorToCVMat(std::get<0>(render_result).Outputs1.DepthMap.detach().cpu()));
		}

	if (ViewParams.DrawNeRFDisp)
		if (std::get<0>(render_result).Outputs1.DispMap.defined())
		{
			SetRenderMat(TorchTensorToCVMat(std::get<0>(render_result).Outputs1.DispMap.cpu()));
		}

	if (ViewParams.DrawLeRF)
		if (std::get<1>(render_result).Outputs1.Relevancy.defined())
		{
			torch::Tensor rel = std::get<1>(render_result).Outputs1.Relevancy.cpu();
			int w = rel.sizes()[1],
				h = rel.sizes()[0];		//this->size().height();
			cv::Mat relevancy_img(h, w, CV_8UC1/*CV_32FC1*/);
			#pragma omp parallel for
			for (int i = 0; i < w; i++)
				for (int j = 0; j < h; j++)
				{
						float lv = rel.index({j,i,0}).item<float>();
						relevancy_img.at<uchar>(j, i) = cv::saturate_cast<uchar>(lv * 255);	//[-1..1] -> [0..255]
				}
			cv::Mat colored_map;
			cv::applyColorMap(relevancy_img, colored_map, cv::COLORMAP_JET);
			SetRenderMat(colored_map);
		}
	update();
}

void TNeRFRenderWidget :: SetLeRFPrompts(const std::string &lerf_positives, const std::vector<std::string> &lerf_negatives)
{
	this->Executor.SetLeRFPrompts(lerf_positives, lerf_negatives);

	update();
}