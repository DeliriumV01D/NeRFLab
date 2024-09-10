#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "CuSHEncoder.h"
#include "CuHashEmbedder.h"
#include "NeRF.h"
#include "BaseNeRFRenderer.h"
#include "NeRFExecutor.h"

#include <TabToolbar/TabToolbar.h>
#include <TabToolbar/Page.h>
#include <TabToolbar/Group.h>
#include <TabToolbar/SubGroup.h>
#include <TabToolbar/StyleTools.h>
#include <TabToolbar/Builder.h>

#include <QPushButton>
#include <QTextEdit>
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

const std::string DATA_DIR = "..//..//NeRF++//data//nerf_synthetic//drums";


#include "TNeRFLabMainWindow.h"
#include "./ui_tnerflabmainwindow.h"

TNeRFLabMainWindow :: TNeRFLabMainWindow(QWidget *parent)
	: QMainWindow(parent), ui(new Ui::TNeRFLabMainWindow)
{
	ui->setupUi(this);

	tt::Builder ttb(this);
	NeRFLabMainWindowToolbar = new tt::TabToolbar(this, 75, 3);
	addToolBar(Qt::TopToolBarArea, NeRFLabMainWindowToolbar);

	tt::Page * page_file = NeRFLabMainWindowToolbar->AddPage(" File ");
	tt::Group * groupFile = page_file->AddGroup("File");
	groupFile->AddAction(QToolButton::DelayedPopup, ui->actionOpen);
	groupFile->AddAction(QToolButton::DelayedPopup, ui->actionSave);
	groupFile->AddAction(QToolButton::DelayedPopup, ui->actionSaveAs);
	connect(ui->actionOpen, SIGNAL(triggered()), this, SLOT(OnActionOpenTriggered()));
	connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(OnActionSaveTriggered()));
	connect(ui->actionSaveAs, SIGNAL(triggered()), this, SLOT(OnActionSaveAsTriggered()));

	tt::Page * page_preprocessing = NeRFLabMainWindowToolbar->AddPage(" Preprocessing ");

	tt::Page * page_training = NeRFLabMainWindowToolbar->AddPage(" Training ");


	tt::Page * page_view = NeRFLabMainWindowToolbar->AddPage(" View ");
	tt::Group * groupViewMode = page_view->AddGroup("Mode");
	QMenu * menuViewMode = new QMenu(this);
	menuViewMode->setObjectName("menuViewMode");
	acNeRFRGB = new QAction("NeRF RGB");
	acNeRFRGB->setCheckable(true);
	acNeRFRGB->setChecked(true);
	menuViewMode->addAction(acNeRFRGB);
	acNeRFDepth = new QAction("NeRF Depth");
	acNeRFDepth->setCheckable(true);
	menuViewMode->addAction(acNeRFDepth);
	acNeRFDisp = new QAction("NeRF Disp");
	acNeRFDisp->setCheckable(true);
	menuViewMode->addAction(acNeRFDisp);
	groupViewMode->AddAction(QToolButton::MenuButtonPopup, ui->actionViewMode, menuViewMode);
	connect(acNeRFRGB, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFRGBTriggered(bool)));
	connect(acNeRFDepth, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFDepthTriggered(bool)));
	connect(acNeRFDisp, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFDispTriggered(bool)));

	tt::Group * groupViewLerf = page_view->AddGroup("LeRF");
	//g2->AddSeparator();
	QTextEdit * tePrompt = new QTextEdit();
	groupViewLerf->AddWidget(tePrompt);
	tePrompt->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Ignored);
	tePrompt->setMaximumWidth(200);

	tt::Group * stylesGroup = page_view->AddGroup("stylesGroup");
	stylesGroup->AddSeparator();
	tt::SubGroup * stylesGroupSub = stylesGroup->AddSubGroup(tt::SubGroup::Align::Yes);
	//create buttons for each style
	QStringList styles = tt::GetRegisteredStyles();
	for(int i=0; i<styles.size(); i++)
	{
		const QString styleName = styles.at(i);
		std::cout<<styleName.toStdString()<<std::endl;
		QPushButton* btn = new QPushButton(styleName, this);
		QObject::connect(btn, &QPushButton::clicked, [styleName, this]() { NeRFLabMainWindowToolbar->SetStyle("NoStyle"); NeRFLabMainWindowToolbar->SetStyle(styleName); });
		stylesGroup->AddWidget(btn);
	}
	tt::RegisterStyle("NoStyle", []()
		{
			tt::StyleParams* params = new tt::StyleParams();
			params->UseTemplateSheet = false;
			params->AdditionalStyleSheet = "";
			return params;
		});
	
	QPushButton * btn = new QPushButton("nativeStyleButton");
	btn->setText("No Style");
	QObject::connect(btn, &QPushButton::clicked, [this]() { NeRFLabMainWindowToolbar->SetStyle("NoStyle"); });
	stylesGroup->AddWidget(btn);
	
	btn = new QPushButton("defaultStyleButton");
	btn->setText("Default");
	QObject::connect(btn, &QPushButton::clicked, [this]() { NeRFLabMainWindowToolbar->SetStyle(tt::GetDefaultStyle()); });
	stylesGroup->AddWidget(btn);

	NeRFLabMainWindowToolbar->SetStyle("NoStyle"); 
	NeRFLabMainWindowToolbar->SetStyle(tt::GetDefaultStyle());



	//QMenu * menu = new QMenu(this);
	//menu->setObjectName("dummyMenu");
	//menu->addActions({ui->actionDummy});

	//tt::Page * page_example = NeRFLabMainWindowToolbar->AddPage(" Example ");
	//tt::Group * g1 = page_example->AddGroup("Group 1");
	//tt::Group * g2 = page_example->AddGroup("Group 2");
	//tt::Group * g3 = page_example->AddGroup("Group 3");
	//g1->AddSeparator();
	//g1->AddAction(QToolButton::DelayedPopup, ui->actionSave);
	//g1->AddAction(QToolButton::DelayedPopup, ui->actionSaveAs);
	//g2->AddAction(QToolButton::InstantPopup, ui->actionPolypaint, menu);
	////g2->AddAction(QToolButton::InstantPopup, ui.actionSetROI, menu);
	//g2->AddSeparator();
	//QTextEdit* te = new QTextEdit();
	//g2->AddWidget(te);
	//te->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Ignored);
	//te->setMaximumWidth(100);
	//tt::SubGroup * g2s = g2->AddSubGroup(tt::SubGroup::Align::Yes);
	//g2s->AddAction(QToolButton::DelayedPopup, ui->actionUndo);
	//g2s->AddAction(QToolButton::DelayedPopup, ui->actionRedo);
	//g2s->AddAction(QToolButton::InstantPopup, ui->actionClose, menu);
	//g3->AddAction(QToolButton::MenuButtonPopup, ui->actionSettings, menu);
	//tt::SubGroup * g3s = g3->AddSubGroup(tt::SubGroup::Align::Yes);
	//g3s->AddHorizontalButtons({{QToolButton::DelayedPopup, ui->actionSave},
	//	{QToolButton::InstantPopup, ui->actionPolypaint, menu},
	//	{QToolButton::MenuButtonPopup, ui->actionSettings, menu}});
	//g3s->AddHorizontalButtons({{QToolButton::DelayedPopup, ui->actionUndo},
	//	{QToolButton::DelayedPopup, ui->actionRedo},
	//	{QToolButton::InstantPopup, ui->actionClose, menu}});
	//QCheckBox* ch = new QCheckBox("Check 1");
	//g3s->AddWidget(ch);

	//g3->AddSeparator();
	//tt::SubGroup * g3ss = g3->AddSubGroup(tt::SubGroup::Align::No);
	///*QPushButton* */btn = new QPushButton(" Edit ");
	//g3ss->AddWidget(btn);
	//g3ss->AddAction(QToolButton::DelayedPopup, ui->actionSaveAs);
	
	tt::Page * pageHelp = NeRFLabMainWindowToolbar->AddPage(" Help ");

	NeRFLabMainWindowToolbar->AddCornerAction(ui->actionHelp);
	
	QApplication::processEvents();
}

TNeRFLabMainWindow :: ~TNeRFLabMainWindow()
{
	delete ui;
}



/*****************************************************************************************/
//SLOTS
/*****************************************************************************************/

void TNeRFLabMainWindow :: OnActionOpenTriggered()
{
	std::string dataset_dir = QFileDialog::getExistingDirectory(
		this,
		tr("Select dataset directory"),
		"..//..//NeRF++//data//nerf_synthetic//drums",
		QFileDialog::ShowDirsOnly
	).toStdString();

	if (dataset_dir.empty())
		return;

	//QStringList files = QFileDialog::getOpenFileNames(
	//	this,
	//	tr("Select files"),
	//	"",
	//	tr("All files (*.*)")
	//);


	//qDebug() << "focal_length: " << focal_length;
	
	try{
		torch::manual_seed(42);

		const std::string base_dir = "output";

		///Пока что каждый раз очищаем рабочую директорию
		if (std::filesystem::exists(base_dir))
		{
			if (std::filesystem::remove_all(base_dir) != static_cast<std::uintmax_t>(-1))
				std::filesystem::create_directories(base_dir);
		}

		NeRFExecutorParams exparams;
		exparams.net_depth = 2;				//layers in network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width = 64;				//channels per layer 256 for classic NeRF, 64 for HashNeRF
		exparams.multires = 10;
		exparams.use_viewdirs = true;	//use full 5D input instead of 3D Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		exparams.calculate_normals = false;
		exparams.use_pred_normal = false;	//whether to use predicted normals
		exparams.use_lerf = false;
		exparams.multires_views = 7;		//log2 of max freq for positional encoding (2D direction)
		exparams.n_importance = 192;		//number of additional fine samples per ray
		exparams.net_depth_fine = 2;		//layers in fine network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width_fine = 64;	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		exparams.num_layers_color = 2;				//for color part of the HashNeRF
		exparams.hidden_dim_color = 64;			//for color part of the HashNeRF
		exparams.num_layers_color_fine = 3;	//for color part of the HashNeRF
		exparams.hidden_dim_color_fine = 64;	//for color part of the HashNeRF
		exparams.num_layers_normals = 2;			//!!!->2
		exparams.hidden_dim_normals = 64;
		exparams.geo_feat_dim = 15;
		exparams.n_levels = 16;
		exparams.n_features_per_level = 2;
		exparams.log2_hashmap_size = 19;		//19
		exparams.base_resolution = 16;
		exparams.finest_resolution = 512;
		exparams.device = torch::kCUDA;
		exparams.learning_rate = 1e-2;		//5e-4 for classic NeRF
		exparams.ft_path = base_dir;//"..//..//NeRF++//build//output";//"..//output";		//"..//..//NeRF++//build//output";
		exparams.n_levels_le = exparams.n_levels/*32*/,																		//for language embedder
		exparams.n_features_per_level_le = 8/*8*/,								//for language embedder
		exparams.log2_hashmap_size_le = 21,									//for language embedder
		exparams.base_resolution_le = exparams.base_resolution,													//for language embedder
		exparams.finest_resolution_le = exparams.finest_resolution,										//for language embedder
		exparams.clip_input_img_size = 336;	//Input RuClip model size
		exparams.num_layers_le = 2;					//Language embedder head params
		exparams.hidden_dim_le = 256;				//Language embedder head params
		exparams.lang_embed_dim = 768;			//Language embedder head params
		exparams.path_to_clip = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336";									//Path to RuClip model
		exparams.	path_to_bpe = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336//bpe.model";			//Path to tokenizer
		exparams.lerf_positives = "stool chair";
		exparams.lerf_negatives = {"object", "things", "texture"};
		//NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall> nerf_executor(exparams);
		//NeRFExecutor <LeRFEmbedder<CuHashEmbedder>, CuSHEncoder, LeRF> nerf_executor(exparams);
		std::unique_ptr<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall>> nerf_executor = std::make_unique<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall>>(exparams);


		NeRFExecutorTrainParams params;
		params.BaseDir = base_dir;//"output";			//where to store ckpts and logs
		params.RenderOnly = false;			//do not optimize, reload weights and render out render_poses path
		params.Ndc = false;							//use normalized device coordinates (set for non-forward facing scenes)
		params.LinDisp = false;					//sampling linearly in disparity rather than depth
		params.NoBatching = true;				//only take random rays from 1 image at a time
		params.TestSkip = true;
		params.Chunk = 1024 * 4;				//number of rays processed in parallel, decrease if running out of memory
		params.NSamples = 64;						//number of coarse samples per ray
		params.NRand = 32 * 32 * 4;			//batch size (number of random rays per gradient step), decrease if running out of memory
		params.PrecorpIters = 0;				//number of steps to train on central crops
		params.NIters = 6100;
		params.LRateDecay = 3;				//exponential learning rate decay (in 1000 steps)  например: 150 - каждые 150000 итераций скорость обучения будет падать в 10 раз
		//logging / saving options
		params.IPrint = 100;						//frequency of console printout and metric loggin
		params.IImg = 500;							//frequency of tensorboard image logging
		params.IWeights = 6000;				//frequency of weight ckpt saving
		params.ITestset = 6000;				//frequency of testset saving
		params.IVideo = 6200;					//frequency of render_poses video saving
		params.ReturnRaw = false;
		params.RenderFactor = 0;
		params.PrecorpFrac = 0.5f;
		params.PyramidClipEmbeddingSaveDir = dataset_dir;			//

		CompactData data = nerf_executor->LoadData(dataset_dir, DatasetType::BLENDER, 
			false,			///load blender synthetic data at 400x400 instead of 800x800
			params.TestSkip,
			false				///set to render synthetic data on a white bkgd (always use for dvoxels)
		);
		///!!!Сделать нормальное копирование, так как эти тензоры заполняются внутри
		float kdata[] = { data.Focal, 0, 0.5f * data.W,
			0, data.Focal, 0.5f * data.H,
			0, 0, 1 };
		data.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32);
		//data.K = GetCalibrationMatrix(data.Focal, data.W, data.H).clone().detach();
		data.BoundingBox = GetBbox3dForObj(data).clone().detach();
		nerf_executor->Train(data, params);

		//nerf_executor->Initialize(nerf_executor->GetParams(), data.BoundingBox);

		RenderParams render_params;
		render_params.NSamples = params.NSamples;						//number of coarse samples per ray
		render_params.NImportance = exparams.n_importance;				//number of additional fine samples per ray
		render_params.Chunk = params.Chunk;					//number of rays processed in parallel, decrease if running out of memory
		render_params.ReturnRaw = false;
		render_params.LinDisp = false;					//sampling linearly in disparity rather than depth
		render_params.Perturb = 0.f;						//0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		render_params.WhiteBkgr = false;					///If True, assume a white background.
		render_params.RawNoiseStd = 0.;
		render_params.Ndc = false;						///If True, represent ray origin, direction in NDC coordinates.
		render_params.Near = data.Near;						///float or array of shape[batch_size].Nearest distance for a ray.
		render_params.Far = data.Far;							///float or array of shape[batch_size].Farthest distance for a ray.
		render_params.UseViewdirs = true;	///If True, use viewing direction of a point in space in model.
		render_params.CalculateNormals = false;
		render_params.UsePredNormal = false;	///whether to use predicted normals
		render_params.ReturnWeights = false;
		render_params.RenderFactor = 0;
		render_params.UseLeRF = false;
		render_params.LangEmbedDim = 768;
		//render_params.LerfPositives = LerfPositives;
		//render_params.LerfNegatives = LerfNegatives;


		TRenderWidgetViewParams vp;
		vp.DrawNeRFRGB = true;
		vp.DrawNeRFDepth = false;
		vp.DrawNeRFDisp = false;
		//!!!Сделать в каждой из этих процедур проверку на зополненность остальных и update
		ui->RenderWidget->SetRenderParams(render_params);
		ui->RenderWidget->SetK(data.K.clone().detach());
		ui->RenderWidget->SetExecutor(nerf_executor);
		ui->RenderWidget->SetDefaultPose(data.Poses[0].clone().detach());
		ui->RenderWidget->SetViewParams(vp);

	} catch (std::exception &e){ 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
}

void TNeRFLabMainWindow :: OnActionSaveTriggered()
{
}

void TNeRFLabMainWindow :: OnActionSaveAsTriggered()
{
}

void TNeRFLabMainWindow :: OnAcNeRFRGBTriggered(bool checked)
{
	if (checked)
	{
		acNeRFDepth->setChecked(false);
		acNeRFDisp->setChecked(false);
	}
	auto view_params = ui->RenderWidget->GetViewParams();
	view_params.DrawNeRFRGB = acNeRFRGB->isChecked();
	view_params.DrawNeRFDepth = acNeRFDepth->isChecked();
	view_params.DrawNeRFDisp = acNeRFDisp->isChecked();
	ui->RenderWidget->SetViewParams(view_params);
}

void TNeRFLabMainWindow :: OnAcNeRFDepthTriggered(bool checked)
{
	if (checked)
	{
		acNeRFRGB->setChecked(false);
		acNeRFDisp->setChecked(false);
	}
	auto view_params = ui->RenderWidget->GetViewParams();
	view_params.DrawNeRFRGB = acNeRFRGB->isChecked();
	view_params.DrawNeRFDepth = acNeRFDepth->isChecked();
	view_params.DrawNeRFDisp = acNeRFDisp->isChecked();
	ui->RenderWidget->SetViewParams(view_params);
}

void TNeRFLabMainWindow :: OnAcNeRFDispTriggered(bool checked)
{
	if (checked)
	{
		acNeRFRGB->setChecked(false);
		acNeRFDepth->setChecked(false);
	}
	auto view_params = ui->RenderWidget->GetViewParams();
	view_params.DrawNeRFRGB = acNeRFRGB->isChecked();
	view_params.DrawNeRFDepth = acNeRFDepth->isChecked();
	view_params.DrawNeRFDisp = acNeRFDisp->isChecked();
	ui->RenderWidget->SetViewParams(view_params);
}
