//! This module provides conversion of a Point structure to a FlatPoint containing just the Id of a point
//! and those of its neighbours.
//! The whole Hnsw structure is then flattened into a Hashtable associating the data ID of a point to
//! its corresponding FlatPoint.   
//! It can be used, for example, when reloading only the graph part of the data to have knowledge
//! of relative proximity of points as described just by their DataId
//!

use hashbrown::HashMap;
use std::cmp::Ordering;

use crate::core::index::hnsw_rs::hnsw;
use hnsw::*;

// an ordering of Neighbour of a Point

impl PartialEq for Neighbour {
    fn eq(&self, other: &Neighbour) -> bool {
        self.distance == other.distance
    } // end eq
}

impl Eq for Neighbour {}

// order points by distance to self.
impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Neighbour) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    } // end cmp
} // end impl PartialOrd

impl Ord for Neighbour {
    fn cmp(&self, other: &Neighbour) -> Ordering {
        if !self.distance.is_nan() && !other.distance.is_nan() {
            self.distance.partial_cmp(&other.distance).unwrap()
        } else {
            panic!("got a NaN in a distance");
        }
    } // end cmp
}

/// a reduced version of point inserted in the Hnsw structure.
/// It contains original id of point as submitted to the struct Hnsw
/// an ordered (by distance) list of neighbours to the point
/// and it position in layers.
#[derive(Clone)]
pub struct FlatPoint {
    /// an id coming from client using hnsw, should identify point uniquely
    origin_id: DataId,
    /// a point id identifying point as stored in our structure
    p_id: PointId,
    /// neighbours info
    neighbours: Vec<Neighbour>,
}

impl FlatPoint {
    /// returns the neighbours orderded by distance.
    pub fn get_neighbours(&self) -> &Vec<Neighbour> {
        &self.neighbours
    }
    /// returns the origin id of the point
    pub fn get_id(&self) -> DataId {
        self.origin_id
    }
    ///
    pub fn get_p_id(&self) -> PointId {
        self.p_id
    }
} // end impl block for FlatPoint

fn flatten_point<T: Clone + Send + Sync>(point: &Point<T>) -> FlatPoint {
    let neighbours = point.get_neighborhood_id();
    // now we flatten neighbours
    let mut flat_neighbours = Vec::<Neighbour>::new();
    for layer in neighbours {
        for neighbour in layer {
            flat_neighbours.push(neighbour);
        }
    }
    flat_neighbours.sort_unstable();

    FlatPoint {
        origin_id: point.get_origin_id(),
        p_id: point.get_point_id(),
        neighbours: flat_neighbours,
    }
} // end of flatten_point

/// A structure providing neighbourhood information of a point stored in the Hnsw structure given its DataId.  
/// The structure uses the [FlatPoint] structure.  
/// This structure can be obtained by FlatNeighborhood::from<&Hnsw<T,D>>
pub struct FlatNeighborhood {
    hash_t: HashMap<DataId, FlatPoint>,
}

impl FlatNeighborhood {
    /// get neighbour of a point given its id.  
    /// The neighbours are sorted in increasing distance from data_id.
    pub fn get_neighbours(&self, p_id: DataId) -> Option<Vec<Neighbour>> {
        let res = self
            .hash_t
            .get(&p_id)
            .map(|point| point.get_neighbours().clone());
        res
    }
} // end impl block for FlatNeighborhood

impl<T: Clone + Send + Sync, D: Distance<T> + Send + Sync> From<&Hnsw<T, D>> for FlatNeighborhood {
    /// extract from the Hnsw strucure a hashtable mapping original DataId into a FlatPoint structure gathering its neighbourhood information.  
    /// Useful after reloading from a dump with T=NoData and D = NoDist as points are then reloaded with neighbourhood information only.
    fn from(hnsw: &Hnsw<T, D>) -> Self {
        let mut hash_t = HashMap::new();
        let mut ptiter = hnsw.get_point_indexation().into_iter();
        //
        loop {
            if let Some(point) = ptiter.next() {
                //    println!("point : {:?}", _point.p_id);
                let res_insert = hash_t.insert(point.get_origin_id(), flatten_point(&point));
                if let Some(old_point) = res_insert {
                    println!("2 points with same origin id {:?}", old_point.origin_id);
                    log::error!("2 points with same origin id {:?}", old_point.origin_id);
                } // end match
            } else {
                break;
            }
        } // end while
        FlatNeighborhood { hash_t }
    }
} // e,d of Fom implementation
