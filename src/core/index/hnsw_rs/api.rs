//! Api for external language.  
//! This file provides a trait to be used as an opaque pointer for C or Julia calls used in file libext.rs

use serde::{de::DeserializeOwned, Serialize};

use crate::core::index::hnsw_rs::hnsw::*;

pub trait AnnT {
    /// type of data vectors
    type Val;
    ///
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize);
    ///
    fn search_neighbours(&self, data: &Vec<Self::Val>, knbn: usize, ef_s: usize) -> Vec<Neighbour>;
    ///
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)>);
    ///
    fn parallel_search_neighbours(
        &self,
        data: &Vec<Vec<Self::Val>>,
        knbn: usize,
        ef_s: usize,
    ) -> Vec<Vec<Neighbour>>;
    ///
    /// dumps a data and graph in 2 files.
    /// Datas are dumped in file filename.hnsw.data and graph in filename.hnsw.graph
    fn file_dump(&self, filename: &String) -> Result<i32, String>;
}

impl<T, D> AnnT for Hnsw<T, D>
where
    T: Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + crate::core::index::hnsw_rs::arrow_ipc::ArrowType,
    D: Distance<T> + Send + Sync,
{
    type Val = T;
    ///
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize) {
        self.insert((data, id));
    }
    ///
    fn search_neighbours(&self, data: &Vec<T>, knbn: usize, ef_s: usize) -> Vec<Neighbour> {
        self.search(data, knbn, ef_s, None)
    }
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)>) {
        self.parallel_insert(data);
    }

    fn parallel_search_neighbours(
        &self,
        data: &Vec<Vec<Self::Val>>,
        knbn: usize,
        ef_s: usize,
    ) -> Vec<Vec<Neighbour>> {
        self.parallel_search(data, knbn, ef_s)
    }
    fn file_dump(&self, filename: &String) -> Result<i32, String> {
        log::debug!("\n in file_dump : {:?}", filename);
        let mut graphname = filename.clone();
        graphname.push_str(".hnsw.graph");
        let buffer = crate::core::index::hnsw_rs::arrow_ipc::dump_arrow_ipc(self)?;
        std::fs::write(&graphname, buffer).map_err(|e| e.to_string())?;
        log::debug!("\n end of dump");
        Ok(1)
    } // end of dump
} // end of impl block AnnT for Hnsw<T,D>

// macro export makes the macro export t the root of the crate
#[macro_export]
macro_rules! mapdist_t(
    ("DistL1")       => ($crate::core::index::hnsw_rs::dist::DistL1);
    ("DistL2")       => ($crate::core::index::hnsw_rs::dist::DistL2);
    ("DistL2")       => ($crate::core::index::hnsw_rs::dist::DistL2);
    ("DistDot")      => ($crate::core::index::hnsw_rs::dist::DistDot);
    ("DistHamming")  => ($crate::core::index::hnsw_rs::dist::DistHamming);
    ("DistJaccard")  => ($crate::core::index::hnsw_rs::dist::DistJaccard);
    ("DistPtr")      => ($crate::core::index::hnsw_rs::dist::DistPtr);
    ("DistLevenshtein") => ($crate::core::index::hnsw_rs::dist::DistLevenshtein);
    ("DistJensenShannon") => ($crate::core::index::hnsw_rs::dist::DistJensenShannon);
    ("DistHellinger") => ($crate::core::index::hnsw_rs::dist::DistHellinger);
    ("DistJeffreys") => ($crate::core::index::hnsw_rs::dist::DistJeffreys);
);
