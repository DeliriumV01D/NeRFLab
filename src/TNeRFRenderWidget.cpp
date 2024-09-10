#include "TNeRFRenderWidget.h"

#include <qevent.h>
#include <QDebug.h>
#include <QPainter.h>
#include <QtCore/QCoreApplication.h>

#include "CommonDefinitions.h"

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

	connect(&Executor, SIGNAL(UpdateResult(RenderResult)), this, SLOT(OnUpdateResult(RenderResult)));
}

TNeRFRenderWidget :: ~TNeRFRenderWidget()
{
}

void TNeRFRenderWidget :: SetExecutor(std::unique_ptr<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall>> &executor)
{
	std::vector<torch::Tensor> splits = torch::split(executor->GetEmbedderBoundingBox(), { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];

	NRWMutex.lock();
	SceneBoundingSphereRadius = (box_max - box_min).norm().item<float>();
	SetDefaultScene();
	NRWMutex.unlock();

	Executor.SetExecutor(executor);

	Render();
}

void TNeRFRenderWidget :: SetDefaultPose(torch::Tensor default_pose)
{
	//Get rotations from pose
	float found_x_rot, found_y_rot, found_z_rot;
	found_y_rot = -torch::asin(default_pose[0][2]).item<float>();
	auto C = cos(found_y_rot);
	if (fabs( C ) > 0.005)												/* ось y зафиксирована? */
	{
		float tx = default_pose[2][2].item<float>() / C;		/* Нет, находим угол поворота вокруг X */
		float ty = -default_pose[1][2].item<float>() / C;
		found_x_rot  = atan2( ty, tx );

		tx = default_pose[0][0].item<float>() / C;												/* находим угол поворота вокруг оси Z */
		ty = -default_pose[0][1].item<float>() / C;
		found_z_rot  = atan2( ty, tx );
	}	else {																			/* ось y зафиксирована */
		found_x_rot  = 0;														/* Устанавливаем вращеине вокруг X на 0 */
		float tx = default_pose[1][1].item<float>();											/* И считаем вращение вокруг Z */
		float ty = default_pose[1][0].item<float>();
		found_z_rot  = atan2( ty, tx );
	}

	NRWMutex.lock();
	XTra = default_pose[0][3].item<float>();
	YTra = default_pose[1][3].item<float>();
	ZTra = default_pose[2][3].item<float>();
	SceneBoundingSphereRadius = sqrt(XTra * XTra + YTra * YTra + ZTra * ZTra);
	XRot = found_x_rot/PI*180;
	YRot = found_y_rot/PI*180;
	ZRot = found_z_rot/PI*180;
	NSca = 1.f;
	std::cout<<"Found rotations: "<<XRot<<" "<<YRot<<" "<<ZRot<<std::endl;
	std::cout<<"Found translations: "<<XTra<<" "<<YTra<<" "<<ZTra<<std::endl;
	NRWMutex.unlock();

	Render();
};

void TNeRFRenderWidget :: SetRenderParams(const RenderParams &params)
{
	NRWMutex.lock();
	RParams = params;
	NRWMutex.unlock();

	Render();
};

TRenderWidgetViewParams TNeRFRenderWidget :: GetViewParams()
{
	TRenderWidgetViewParams result;
	NRWMutex.lock();
	result = ViewParams;
	NRWMutex.unlock();
	return result;
};

void TNeRFRenderWidget :: SetViewParams(const TRenderWidgetViewParams &params)
{
	NRWMutex.lock();
	ViewParams = params;
	NRWMutex.unlock();

	Render();
};

torch::Tensor TNeRFRenderWidget :: GetRenderPose()
{
	torch::NoGradGuard no_grad;

	NRWMutex.lock();
	// загружаем единичную матрицу моделировани
	float pose_data[] = { 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1};
	auto pose = torch::from_blob(pose_data, { 4, 4 });
	try {
		// масштабирование
		float scale_data[] = { NSca, 0, 0, 0,
			0, NSca, 0, 0,
			0, 0, NSca, 0,
			0, 0, 0, 1};
		auto scale = torch::from_blob(scale_data, { 4, 4 });
		pose = torch::matmul(pose, scale);

		// повороты
		pose = torch::matmul(pose, AxisAngle(torch::tensor({0, 0, 1}), torch::tensor({ZRot/180*PI})) );  // поворот вокруг оси Z
		pose = torch::matmul(pose, AxisAngle(torch::tensor({0, 1, 0}), torch::tensor({YRot/180*PI})) );  // поворот вокруг оси Y
		pose = torch::matmul(pose, AxisAngle(torch::tensor({1, 0, 0}), torch::tensor({XRot/180*PI})) );  // поворот вокруг оси X

		// трансляция
		float trans_data[] = { 1, 0, 0, XTra,
			0, 1, 0, -YTra,
			0, 0, 1, ZTra,
			0, 0, 0, 1};
		auto trans = torch::from_blob(trans_data, { 4, 4 });
		pose = torch::matmul(pose, trans);
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

void TNeRFRenderWidget :: mouseMoveEvent(QMouseEvent * evnt)
{
	NRWMutex.lock();
	if (MouseButtonPressed == Qt::RightButton)
	{
		XRot -= 180 /*/ NSca*/ * (float)(evnt->y() - MousePosition.y()) / height();
		ZRot -= 180 /*/ NSca*/ * (float)(evnt->x() - MousePosition.x()) / width();
	}

	if (MouseButtonPressed == Qt::LeftButton)
	{
		XTra -= (float)(evnt->x() - MousePosition.x())/std::min(width(), height())/NSca*2*SceneBoundingSphereRadius;
		YTra -= (float)(evnt->y() - MousePosition.y())/std::min(width(), height())/NSca*2*SceneBoundingSphereRadius;		
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
	if ((evnt->angleDelta().y()) > 0) ScaleMinus();
	else if ((evnt->angleDelta().y()) < 0) ScalePlus();
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
	ZRot += 1.0f;
}

void TNeRFRenderWidget :: RotateRight()
{
	ZRot -= 1.0f;
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

void TNeRFRenderWidget :: OnUpdateResult(RenderResult render_result)
{
	if (ViewParams.DrawNeRFRGB)
	{
		SetRenderMat(TorchTensorToCVMat(render_result.Outputs1.RGBMap.cpu()));
	}

	if (ViewParams.DrawNeRFDepth)
	{
		render_result.Outputs1.DepthMap = (render_result.Outputs1.DepthMap - RParams.Near) / (RParams.Far - RParams.Near);
		SetRenderMat(TorchTensorToCVMat(render_result.Outputs1.DepthMap.detach().cpu()));
	}

	if (ViewParams.DrawNeRFDisp)
	{
		SetRenderMat(TorchTensorToCVMat(render_result.Outputs1.DispMap.cpu()));
	}
	
	update();

	////rgbs.push_back(render_result.Outputs1.RGBMap.cpu());
	////disps.push_back(render_result.Outputs1.DispMap.cpu());
	////normalize depth to[0, 1]
	//render_result.Outputs1.DepthMap = (render_result.Outputs1.DepthMap - near) / (far - near);

	////torch::Tensor normals_from_depth = NormalMapFromDepthMap(render_result.Outputs1.DepthMap.detach().cpu());
	////if (!savedir.empty())
	////	cv::imwrite((savedir / ("normals_from_depth_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(normals_from_depth));

	//if (!savedir.empty())
	//{
	//	cv::imwrite((savedir / (std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RGBMap.cpu()));
	//	cv::imwrite((savedir / ("disp_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DispMap));
	//	cv::imwrite((savedir / ("depth_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DepthMap));
	//	if (calculate_normals)
	//	{
	//		cv::imwrite((savedir / ("rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedNormals));
	//	}
	//	if (use_pred_normal)
	//	{
	//		cv::imwrite((savedir / ("pred_rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedPredNormals));
	//	}
	//}
}