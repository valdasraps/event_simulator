#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <fstream>
#include <string>

class GenParticleDumper : public edm::one::EDAnalyzer<> {
public:
  explicit GenParticleDumper(const edm::ParameterSet& iConfig)
    : inputTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      fileName_(iConfig.getParameter<std::string>("fileName")),
      token_(consumes<std::vector<reco::GenParticle>>(inputTag_)) {

    out_.open(fileName_);
    if (out_.tellp() == 0) {  
      out_ << "event,pdg_id,status,px,py,pz,pt,eta,phi,energy,mass\n";
    }
  }

  ~GenParticleDumper() override {
    if (out_.is_open()) out_.close();
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup&) override {
    edm::Handle<std::vector<reco::GenParticle>> hGenParticles;
    iEvent.getByToken(token_, hGenParticles);
    int event = iEvent.id().event();

    for (const auto& p : *hGenParticles) {
      out_ << event << "," 
           << p.pdgId() << "," 
           << p.status() << "," 
           << p.px() << ","
           << p.py() << ","
           << p.pz() << ","
           << p.pt() << "," 
           << p.eta() << "," 
           << p.phi() << "," 
           << p.energy() << ","
           << p.mass() << "\n";
    }
  }

private:
  edm::InputTag inputTag_;
  std::string fileName_;
  std::ofstream out_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> token_;
};

DEFINE_FWK_MODULE(GenParticleDumper);
