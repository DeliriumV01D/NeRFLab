//#include "TorchHeader.h"
//#define TT_BUILDING_DLL
#include "TNeRFLabMainWindow.h"
// #include "TRandomDouble.h"
// #include "TRandomInt.h"

#include <QtWidgets/QApplication>
#include <QDebug.h>
#include <typeinfo>

int main(int argc, char * argv[]) 
{
	// TRandomInt::Instance().Initialize((unsigned long)std::time(0));
	// TRandomDouble::Instance().Initialize(RandomInt());

	QApplication application(argc, argv);
	TNeRFLabMainWindow nerf_lab_main_window;
	nerf_lab_main_window.showMaximized();
	return application.exec();
}