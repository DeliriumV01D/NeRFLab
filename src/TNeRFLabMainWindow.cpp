#include "TorchHeader.h"
#include "Trainable.h"
#include "CuSHEncoder.h"
#include "CuHashEmbedder.h"
#include "NeRF.h"
#include "NeRFRenderer.h"
#include "NeRFExecutor.h"

#include "ColmapReconstruction.h"

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
const std::string BASE_DIR = "output";
const std::string CMDB_DIR = "cmdb";


#include "TNeRFLabMainWindow.h"
#include "./ui_tnerflabmainwindow.h"

TNeRFLabMainWindow :: TNeRFLabMainWindow(QWidget *parent)
	: QMainWindow(parent), ui(new Ui::TNeRFLabMainWindow)
{
	ui->setupUi(this);

	tt::Builder ttb(this);
	NeRFLabMainWindowToolbar = new tt::TabToolbar(this, 75, 3);
	addToolBar(Qt::TopToolBarArea, NeRFLabMainWindowToolbar);

	tt::Page * page_data = NeRFLabMainWindowToolbar->AddPage(" Data ");
	tt::Group * groupData = page_data->AddGroup("Data");

	QMenu * menuOpen = new QMenu(this);
	menuOpen->setObjectName(" menuOpen ");
	menuOpen->addAction(ui->acOpenImageFolder);
	menuOpen->addAction(ui->acOpenBlenderDataset);
	menuOpen->addAction(ui->acOpenColmapReconstruction);
	groupData->AddAction(QToolButton::MenuButtonPopup, ui->actionOpen, menuOpen);
	connect(ui->acOpenImageFolder, SIGNAL(triggered(bool)), this, SLOT(OnAcOpenImageFolderTriggered()));
	connect(ui->acOpenBlenderDataset, SIGNAL(triggered(bool)), this, SLOT(OnAcOpenBlenderDatasetTriggered()));
	connect(ui->acOpenColmapReconstruction, SIGNAL(triggered(bool)), this, SLOT(OnAcOpenColmapReconstructionTriggered()));
	
	groupData->AddAction(QToolButton::DelayedPopup, ui->actionSave);
	groupData->AddAction(QToolButton::DelayedPopup, ui->actionSaveAs);
	connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(OnActionSaveTriggered()));
	connect(ui->actionSaveAs, SIGNAL(triggered()), this, SLOT(OnActionSaveAsTriggered()));

	tt::Page * page_process = NeRFLabMainWindowToolbar->AddPage(" Process ");

	tt::Page * page_nerf = NeRFLabMainWindowToolbar->AddPage(" NeRF ");
	tt::Group * groupNerf = page_nerf->AddGroup("NeRF");
	groupNerf->AddAction(QToolButton::DelayedPopup, ui->actionOpenNerf);
	groupNerf->AddAction(QToolButton::DelayedPopup, ui->actionSaveNerf);
	groupNerf->AddAction(QToolButton::DelayedPopup, ui->actionTrainNerf);
	groupNerf->AddAction(QToolButton::DelayedPopup, ui->actionTrainLerf);
	connect(ui->actionOpenNerf, SIGNAL(triggered()), this, SLOT(OnActionOpenNerfTriggered()));
	connect(ui->actionSaveNerf, SIGNAL(triggered()), this, SLOT(OnActionSaveNerfTriggered()));
	connect(ui->actionTrainNerf, SIGNAL(triggered()), this, SLOT(OnActionTrainNerfTriggered()));
	connect(ui->actionTrainLerf, SIGNAL(triggered()), this, SLOT(OnActionTrainLerfTriggered()));

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
	acLeRF = new QAction("LeRF");
	acLeRF->setCheckable(true);
	menuViewMode->addAction(acLeRF);
	groupViewMode->AddAction(QToolButton::MenuButtonPopup, ui->actionViewMode, menuViewMode);
	connect(acNeRFRGB, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFRGBTriggered(bool)));
	connect(acNeRFDepth, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFDepthTriggered(bool)));
	connect(acNeRFDisp, SIGNAL(triggered(bool)), this, SLOT(OnAcNeRFDispTriggered(bool)));
	connect(acLeRF, SIGNAL(triggered(bool)), this, SLOT(OnAcLeRFTriggered(bool)));

	tt::Group * groupViewLerf = page_view->AddGroup("LeRF");
	//g2->AddSeparator();
	tePrompt = new QTextEdit();
	groupViewLerf->AddWidget(tePrompt);
	tePrompt->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Ignored);
	tePrompt->setMaximumWidth(200);
	connect(tePrompt, SIGNAL(textChanged()), this, SLOT(OnTePromptTextChanged()));
	

	tt::Group * stylesGroup = page_view->AddGroup("stylesGroup");
	stylesGroup->AddSeparator();
	tt::SubGroup * stylesGroupSub = stylesGroup->AddSubGroup(tt::SubGroup::Align::Yes);
	//create buttons for each style
	QStringList styles = tt::GetRegisteredStyles();
	for (int i = 0; i < styles.size(); i++)
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


void TNeRFLabMainWindow :: OnAcOpenImageFolderTriggered()
{
	auto image_dir = QFileDialog::getExistingDirectory(
		this,
		tr("Select image directory"),
		"..//..//NeRF++//data//nerf_synthetic//drums",
		QFileDialog::ShowDirsOnly
	).toStdString();

	if (image_dir.empty())
		return;

	try{
		///Пока что каждый раз очищаем рабочую директорию
		if (std::filesystem::exists(CMDB_DIR))
			if (std::filesystem::remove_all(CMDB_DIR) != static_cast<std::uintmax_t>(-1));
		std::filesystem::create_directories(CMDB_DIR);

		ColmapReconstruction(image_dir, CMDB_DIR);

		Data = LoadDatasetParams(
			CMDB_DIR,
			torch::kCUDA,
			DatasetType::COLMAP,
			false,			///load blender synthetic data at 400x400 instead of 800x800
			true,
			false				///set to render synthetic data on a white bkgd (always use for dvoxels)
		);

	} catch (std::exception &e) { 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
}

void TNeRFLabMainWindow :: OnAcOpenBlenderDatasetTriggered()
{
	DatasetDir = QFileDialog::getExistingDirectory(
		this,
		tr("Select dataset directory"),
		"..//..//NeRF++//data//nerf_synthetic//drums",
		QFileDialog::ShowDirsOnly
	).toStdString();

	if (DatasetDir.empty())
		return;

	try{
		torch::manual_seed(42);

		Data = LoadDatasetParams(
			DatasetDir,
			torch::kCUDA,
			DatasetType::BLENDER,
			false,			///load blender synthetic data at 400x400 instead of 800x800
			true,
			false				///set to render synthetic data on a white bkgd (always use for dvoxels)
		);

	} catch (std::exception &e){ 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
}

void TNeRFLabMainWindow :: OnAcOpenColmapReconstructionTriggered()
{
	auto cmdb_dir = QFileDialog::getExistingDirectory(
		this,
		tr("Select COLMAP workspace directory"),
		"cmdb",
		QFileDialog::ShowDirsOnly
	).toStdString();

	if (cmdb_dir.empty())
		return;

	try{
		Data = LoadDatasetParams(
			cmdb_dir,
			torch::kCUDA,
			DatasetType::COLMAP,
			false,			///load blender synthetic data at 400x400 instead of 800x800
			true,
			false				///set to render synthetic data on a white bkgd (always use for dvoxels)
		);
	} catch (std::exception &e) { 
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

void TNeRFLabMainWindow :: OnActionOpenNerfTriggered()
{
	NeRFDir = QFileDialog::getExistingDirectory(
		this,
		tr("Select NeRF directory"),
		QString::fromLocal8Bit(BASE_DIR),
		QFileDialog::ShowDirsOnly
	).toStdString();

	if (NeRFDir.empty())
		return;

	try{
		torch::manual_seed(42);

		NeRFExecutorParams exparams;
		exparams.LoadFromFile(std::filesystem::path(NeRFDir) / "executor_params.json");
		std::unique_ptr<TThreadedNeRFExecutor::TExecutor> nerf_executor = std::make_unique<TThreadedNeRFExecutor::TExecutor>(exparams);

		NeRFExecutorTrainParams params;
		params.LoadFromFile(std::filesystem::path(NeRFDir) / "executor_train_params.json");

		Data.LoadFromFile(std::filesystem::path(NeRFDir) / "data.json");

		nerf_executor->Initialize(exparams, Data.BoundingBox);
		if (exparams.use_lerf)
			nerf_executor->SetLeRFPrompts(exparams.lerf_positives, exparams.lerf_negatives);//nerf_executor->InitializeTestLeRF(params, Data);


		//reset_near_plane: whether to reset the near plane to 0.0 during inference.The near plane can be
		//helpful for reducing floaters during training, but it can cause clipping artifacts during
		//inference when an evaluation or viewer camera moves closer to the object.
		std::unique_ptr<NeRFRenderParams> render_params(nerf_executor->FillRenderParams(nerf_executor->GetParams(), params, 0.f/*Data.Near*/, Data.Far, std::numeric_limits<int>::max(), Data.BoundingBox, false, nerf_executor->GetParams().calculate_normals || nerf_executor->GetParams().use_pred_normal));

		TRenderWidgetViewParams vp;
		vp.DrawNeRFRGB = true;
		vp.DrawNeRFDepth = false;
		vp.DrawNeRFDisp = false;
		vp.DrawLeRF = exparams.use_lerf;
		//!!!Сделать в каждой из этих процедур проверку на зополненность остальных и update
		ui->RenderWidget->SetRenderParams(*render_params, false);
		std::cout << "render K: " << GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()) << std::endl;
		ui->RenderWidget->SetK(GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()));
		ui->RenderWidget->SetExecutor(nerf_executor, false);
		ui->RenderWidget->SetDefaultPose(Data.Poses[0].clone().detach(), false);
		ui->RenderWidget->SetViewParams(vp, true);

	} catch (std::exception &e){ 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
}

void TNeRFLabMainWindow :: OnActionSaveNerfTriggered()
{
}

void TNeRFLabMainWindow :: OnActionTrainNerfTriggered()
{
	try{
		torch::manual_seed(42);

		///Пока что каждый раз очищаем рабочую директорию
		if (std::filesystem::exists(BASE_DIR))
			std::filesystem::remove_all(BASE_DIR) != static_cast<std::uintmax_t>(-1);
		std::filesystem::create_directories(BASE_DIR);

		NeRFExecutorParams exparams;
		exparams.net_depth = 2;				//layers in network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width = 64;				//channels per layer 256 for classic NeRF, 64 for HashNeRF
		exparams.multires = 10;
		exparams.use_nerf = true;
		exparams.use_viewdirs = true;	//use full 5D input instead of 3D Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		exparams.calculate_normals = false;
		exparams.use_pred_normal = false;	//whether to use predicted normals
		exparams.use_lerf = false;
		exparams.multires_views = 8;		//log2 of max freq for positional encoding (2D direction)
		exparams.n_importance = 192;		//number of additional fine samples per ray
		exparams.net_depth_fine = 3;		//layers in fine network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width_fine = 64;	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		exparams.num_layers_color = 2;				//for color part of the HashNeRF
		exparams.hidden_dim_color = 64;			//for color part of the HashNeRF
		exparams.num_layers_color_fine = 3;	//for color part of the HashNeRF
		exparams.hidden_dim_color_fine = 64;	//for color part of the HashNeRF
		exparams.num_layers_normals = 2;			//!!!->2
		exparams.hidden_dim_normals = 64;
		exparams.geo_feat_dim = 15;
		exparams.n_levels = 18;
		exparams.n_features_per_level = 2;
		exparams.log2_hashmap_size = 21;		//19
		exparams.base_resolution = 16;
		exparams.finest_resolution = 1024;
		exparams.device = torch::kCUDA;
		exparams.learning_rate = 1e-2;		//5e-4 for classic NeRF
		exparams.ft_path = BASE_DIR;//"..//..//NeRF++//build//output";//"..//output";		//"..//..//NeRF++//build//output";
		exparams.n_levels_le = exparams.n_levels/*32*/,																		//for language embedder
		exparams.n_features_per_level_le = 8/*8*/,								//for language embedder
		exparams.log2_hashmap_size_le = 21,									//for language embedder
		exparams.base_resolution_le = exparams.base_resolution,													//for language embedder
		exparams.finest_resolution_le = exparams.finest_resolution,										//for language embedder
		exparams.clip_input_img_size = 336;	//Input RuClip model size
		exparams.num_layers_le = 2;					//Language embedder head params
		exparams.hidden_dim_le = 256;				//Language embedder head params
		exparams.lang_embed_dim = 768;			//Language embedder head params
		exparams.geo_feat_dim_le = 32;			//Language embedder head params
		exparams.pyr_embed_min_zoom_out = 0;
		exparams.pyr_embedder_overlap = 0.5;
		exparams.path_to_clip = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336";									//Path to RuClip model
		exparams.	path_to_bpe = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336//bpe.model";			//Path to tokenizer
		exparams.lerf_positives = "stool chair";
		exparams.lerf_negatives = {"object", "things", "texture"};
		//NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall> nerf_executor(exparams);
		//NeRFExecutor <LeRFEmbedder<CuHashEmbedder>, CuSHEncoder, LeRF> nerf_executor(exparams);
		std::unique_ptr<TThreadedNeRFExecutor::TExecutor> nerf_executor = std::make_unique<TThreadedNeRFExecutor::TExecutor>(exparams);


		NeRFExecutorTrainParams params;
		params.BaseDir = BASE_DIR;//"output";			//where to store ckpts and logs
		params.RenderOnly = false;			//do not optimize, reload weights and render out render_poses path
		params.Ndc = false;							//use normalized device coordinates (set for non-forward facing scenes)
		params.LinDisp = false;					//sampling linearly in disparity rather than depth
		params.TestSkip = true;
		params.Chunk = 1024 * (exparams.use_lerf ? 1 : 4);				//number of rays processed in parallel, decrease if running out of memory, <= NRand
		params.NSamples = 64;						//number of coarse samples per ray
		params.NRand = 32 * 32 * (exparams.use_lerf ? 1 : 16);			//batch size (number of random rays per gradient step), decrease if running out of memory, >= Chunk, n*Chunk
		params.PrecorpIters = 0;				//number of steps to train on central crops
		params.IPrint = 100;						//frequency of console printout and metric loggin
		params.NIters = powf(float(Data.W) / 800 * float(Data.H) / 800, 0.7f) * 6000 + params.IPrint;
		params.LRateDecay = float(params.NIters)/1000 * 2/3;				//exponential learning rate decay (in 1000 steps)  например: 150 - каждые 150000 итераций скорость обучения будет падать в 10 раз
		//logging / saving options
		params.IImg = 500;							//frequency of tensorboard image logging
		params.IWeights = params.NIters - params.IPrint;				//frequency of weight ckpt saving
		params.ITestset = params.NIters - params.IPrint;				//frequency of testset saving
		params.IVideo = params.NIters + params.IPrint;					//frequency of render_poses video saving
		params.ReturnRaw = false;
		params.RenderFactor = 0;
		params.PrecorpFrac = 0.5f;
		params.PyramidClipEmbeddingSaveDir = DatasetDir;			//

		nerf_executor->Train(Data, params);

		exparams.SaveToFile(params.BaseDir / "executor_params.json");
		params.SaveToFile(params.BaseDir / "executor_train_params.json");
		Data.SaveToFile(params.BaseDir / "data.json");

		//reset_near_plane: whether to reset the near plane to 0.0 during inference.The near plane can be
		//helpful for reducing floaters during training, but it can cause clipping artifacts during
		//inference when an evaluation or viewer camera moves closer to the object.
		std::unique_ptr<NeRFRenderParams> render_params(nerf_executor->FillRenderParams(nerf_executor->GetParams(), params, 0./*Data.Near*/, Data.Far, std::numeric_limits<int>::max(), Data.BoundingBox, false, nerf_executor->GetParams().calculate_normals || nerf_executor->GetParams().use_pred_normal));

		TRenderWidgetViewParams vp;
		vp.DrawNeRFRGB = true;
		vp.DrawNeRFDepth = false;
		vp.DrawNeRFDisp = false;
		vp.DrawLeRF = false;
		//!!!Сделать в каждой из этих процедур проверку на зополненность остальных и update
		ui->RenderWidget->SetRenderParams(*render_params, false);
		std::cout << "render K: " << GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()) << std::endl;
		ui->RenderWidget->SetK(GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()));
		ui->RenderWidget->SetExecutor(nerf_executor, false);
		ui->RenderWidget->SetDefaultPose(Data.Poses[0].clone().detach(), false);
		ui->RenderWidget->SetViewParams(vp, true);

	} catch (std::exception &e) { 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
}

void TNeRFLabMainWindow :: OnActionTrainLerfTriggered()
{
	try {
		torch::manual_seed(42);

		///Пока что каждый раз очищаем рабочую директорию
		if (std::filesystem::exists(BASE_DIR))
			std::filesystem::remove_all(BASE_DIR) != static_cast<std::uintmax_t>(-1);
		std::filesystem::create_directories(BASE_DIR);

		NeRFExecutorParams exparams;
		exparams.net_depth = 2;				//layers in network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width = 64;				//channels per layer 256 for classic NeRF, 64 for HashNeRF
		exparams.multires = 10;
		exparams.use_nerf = false;
		exparams.use_viewdirs = false;	//use full 5D input instead of 3D Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		exparams.calculate_normals = false;
		exparams.use_pred_normal = false;	//whether to use predicted normals
		exparams.use_lerf = true;
		exparams.multires_views = 8;		//log2 of max freq for positional encoding (2D direction)
		exparams.n_importance = 192;		//number of additional fine samples per ray
		exparams.net_depth_fine = 3;		//layers in fine network 8 for classic NeRF, 2/3 for HashNeRF
		exparams.net_width_fine = 64;	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		exparams.num_layers_color = 2;				//for color part of the HashNeRF
		exparams.hidden_dim_color = 64;			//for color part of the HashNeRF
		exparams.num_layers_color_fine = 3;	//for color part of the HashNeRF
		exparams.hidden_dim_color_fine = 64;	//for color part of the HashNeRF
		exparams.num_layers_normals = 2;			//!!!->2
		exparams.hidden_dim_normals = 64;
		exparams.geo_feat_dim = 15;
		exparams.n_levels = 18;
		exparams.n_features_per_level = 2;
		exparams.log2_hashmap_size = 21;		//19
		exparams.base_resolution = 16;
		exparams.finest_resolution = 1024;
		exparams.device = torch::kCUDA;
		exparams.learning_rate = 1e-2;		//5e-4 for classic NeRF
		exparams.ft_path = BASE_DIR;//"..//..//NeRF++//build//output";//"..//output";		//"..//..//NeRF++//build//output";
		exparams.n_levels_le = exparams.n_levels/*32*/,																		//for language embedder
		exparams.n_features_per_level_le = 8/*8*/,								//for language embedder
		exparams.log2_hashmap_size_le = 21,									//for language embedder
		exparams.base_resolution_le = exparams.base_resolution,													//for language embedder
		exparams.finest_resolution_le = exparams.finest_resolution,										//for language embedder
		exparams.pyr_embed_min_zoom_out = 0;
		exparams.pyr_embedder_overlap = 0.5f;
		exparams.clip_input_img_size = 336;	//Input RuClip model size
		exparams.num_layers_le = 2;					//Language embedder head params
		exparams.hidden_dim_le = 256;				//Language embedder head params
		exparams.lang_embed_dim = 768;			//Language embedder head params
		exparams.geo_feat_dim_le = 32;			//Language embedder head params
		exparams.path_to_clip = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336";									//Path to RuClip model
		exparams.path_to_bpe = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336//bpe.model";			//Path to tokenizer
		exparams.lerf_positives = "stool chair";
		exparams.lerf_negatives = {"object", "things", "texture"};
	
		std::unique_ptr<TThreadedNeRFExecutor::TExecutor> nerf_executor = std::make_unique<TThreadedNeRFExecutor::TExecutor>(exparams);

		NeRFExecutorTrainParams params;
		params.BaseDir = BASE_DIR;//"output";			//where to store ckpts and logs
		params.RenderOnly = false;			//do not optimize, reload weights and render out render_poses path
		params.Ndc = false;							//use normalized device coordinates (set for non-forward facing scenes)
		params.LinDisp = false;					//sampling linearly in disparity rather than depth
		params.TestSkip = true;
		params.Chunk = 1024 * (exparams.use_lerf ? 1 : 4);				//number of rays processed in parallel, decrease if running out of memory, <= NRand
		params.NSamples = 64;						//number of coarse samples per ray
		params.NRand = 32 * 32 * (exparams.use_lerf ? 1 : 16);			//batch size (number of random rays per gradient step), decrease if running out of memory, >= Chunk, n*Chunk
		params.PrecorpIters = 0;				//number of steps to train on central crops
		params.IPrint = 100;						//frequency of console printout and metric loggin
		params.NIters = powf(float(Data.W) / 800 * float(Data.H) / 800, 0.7f) * 6000 + params.IPrint;
		params.LRateDecay = float(params.NIters) / 1000 * 2 / 3;				//exponential learning rate decay (in 1000 steps)  например: 150 - каждые 150000 итераций скорость обучения будет падать в 10 раз
		//logging / saving options
		params.IImg = 500;							//frequency of tensorboard image logging
		params.IWeights = params.NIters - params.IPrint;				//frequency of weight ckpt saving
		params.ITestset = params.NIters - params.IPrint;				//frequency of testset saving
		params.IVideo = params.NIters + params.IPrint;					//frequency of render_poses video saving
		params.ReturnRaw = false;
		params.RenderFactor = 0;
		params.PrecorpFrac = 0.5f;
		params.PyramidClipEmbeddingSaveDir = DatasetDir;			//

		nerf_executor->Train(Data, params);

		//nerf_executor->Initialize(nerf_executor->GetParams(), data.BoundingBox);

		exparams.SaveToFile(params.BaseDir / "executor_params.json");
		params.SaveToFile(params.BaseDir / "executor_train_params.json");
		Data.SaveToFile(params.BaseDir / "data.json");


		//reset_near_plane: whether to reset the near plane to 0.0 during inference.The near plane can be
		//helpful for reducing floaters during training, but it can cause clipping artifacts during
		//inference when an evaluation or viewer camera moves closer to the object.
		std::unique_ptr<NeRFRenderParams> render_params(nerf_executor->FillRenderParams(nerf_executor->GetParams(), params, 0.f/*Data.Near*/, Data.Far, std::numeric_limits<int>::max(), Data.BoundingBox, false, nerf_executor->GetParams().calculate_normals || nerf_executor->GetParams().use_pred_normal));

		TRenderWidgetViewParams vp;
		vp.DrawNeRFRGB = false;
		vp.DrawNeRFDepth = false;
		vp.DrawNeRFDisp = false;
		vp.DrawLeRF = true;
		//!!!Сделать в каждой из этих процедур проверку на зополненность остальных и update
		ui->RenderWidget->SetRenderParams(*render_params, false);
		std::cout << "render K: " << GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()) << std::endl;
		ui->RenderWidget->SetK(GetSameFOVCalibrationMatrix(Data.K.clone().detach(), ui->RenderWidget->size().width(), ui->RenderWidget->size().height()));
		ui->RenderWidget->SetExecutor(nerf_executor, false);
		ui->RenderWidget->SetDefaultPose(Data.Poses[0].clone().detach(), false);
		ui->RenderWidget->SetViewParams(vp, true);

	} catch (std::exception &e) { 
		std::cout<<e.what()<<std::endl;
		int ret = QMessageBox::critical(this, tr("NeRFLab"),
			tr(": \n") + tr(e.what()),
			QMessageBox::Ok
		);
	}
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
	view_params.DrawLeRF = acLeRF->isChecked();
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
	view_params.DrawLeRF = acLeRF->isChecked();
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
	view_params.DrawLeRF = acLeRF->isChecked();
	ui->RenderWidget->SetViewParams(view_params);
}

void TNeRFLabMainWindow :: OnAcLeRFTriggered(bool checked)
{
	auto view_params = ui->RenderWidget->GetViewParams();
	view_params.DrawNeRFRGB = acNeRFRGB->isChecked();
	view_params.DrawNeRFDepth = acNeRFDepth->isChecked();
	view_params.DrawNeRFDisp = acNeRFDisp->isChecked();
	view_params.DrawLeRF = acLeRF->isChecked();
	ui->RenderWidget->SetViewParams(view_params);
}

void TNeRFLabMainWindow :: OnTePromptTextChanged()
{
	//Отсчитываем секунду на следующее изменение промпта и стартуем
	ui->RenderWidget->SetLeRFPrompts(tePrompt->toPlainText().toStdString(), {"object", "things", "texture"});
}
