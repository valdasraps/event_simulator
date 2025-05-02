void plotSimHits3D(const char* filename = "psimhits_all.csv", int maxHits = 100) {
    // Load required libraries
    gSystem->Load("libTree");
    gSystem->Load("libGeom");
    gSystem->Load("libEve");

    // Create canvas and 3D viewer
    auto c = new TCanvas("c", "SimHits 3D", 800, 600);
    auto eve = new TEveManager(800, 600); // width, height
    auto geom = new TGeoManager("geom", "Geometry for hits");

    // Open CSV as text file
    std::ifstream infile(filename);
    std::string line;

    // Skip the header line
    std::getline(infile, line);

    int hitCount = 0;

    while (std::getline(infile, line)) {
        if (maxHits > 0 && hitCount >= maxHits) break;

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 13) continue;

        float entryX = std::stof(tokens[4]);
        float entryY = std::stof(tokens[5]);
        float entryZ = std::stof(tokens[6]);
        float exitX  = std::stof(tokens[7]);
        float exitY  = std::stof(tokens[8]);
        float exitZ  = std::stof(tokens[9]);

        std::cout << "Hit line added: (" << entryX << "," << entryY << "," << entryZ << ") -> (" << exitX << "," << exitY << "," << exitZ << ")" << std::endl;

        auto line3D = new TEveLine();
        line3D->SetLineColor(kBlue);
        line3D->SetLineWidth(1);
        line3D->SetNextPoint(entryX, entryY, entryZ);
        line3D->SetNextPoint(exitX, exitY, exitZ);
        // eve->AddElement(line3D);
        gEve->AddElement(line3D);

        ++hitCount;
    }

    auto axes = new TGeoBBox(100, 100, 100); // dummy geometry box
    auto volume = new TGeoVolume("box", axes);
    geom->SetTopVolume(volume);
    geom->CloseGeometry();

    // eve->Redraw3D();
    gEve->FullRedraw3D(kTRUE);

}

