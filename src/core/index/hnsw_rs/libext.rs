#![allow(clippy::missing_safety_doc)]
//! This file contains lib to call hnsw from julia (or any language providing a C api)
//! The AnnT trait is implemented with macros for u32, u16, u8, f32, f64 and i32.  
//! The macro declare_myapi_type!  produces struct HnswApif32 and so on.

use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::PathBuf;
use std::ptr;

use log;

use crate::core::index::hnsw_rs::api::*;
use crate::core::index::hnsw_rs::hnsw::*;
use crate::core::index::hnsw_rs::hnswio::*;
use crate::core::index::hnsw_rs::*;

/// The export macro makes the macro global in crate and accessible via crate::core::index::hnsw_rs::declare_myapi_type!
#[macro_export]
macro_rules! declare_myapi_type(
    ($name:ident, $ty:ty) => (
        pub struct $name {
            #[allow(dead_code)]
            pub(crate) opaque: Box<dyn AnnT<Val=$ty>>,
        }
        impl $name {
            pub fn new(arg: Box<dyn AnnT<Val=$ty>>) -> Self {
                $name{ opaque:arg}
            }
        }
    )
);

declare_myapi_type!(HnswApiNodata, NoData);
declare_myapi_type!(HnswApif64, f64);
declare_myapi_type!(HnswApif32, f32);
declare_myapi_type!(HnswApii32, i32);
declare_myapi_type!(HnswApiu32, u32);
declare_myapi_type!(HnswApiu16, u16);
declare_myapi_type!(HnswApiu8, u8);

/// To be able to return a vector from rust in a julia struct before converting to a julia Vector
#[repr(C)]
pub struct Vec_api<T> {
    pub len: i64,
    pub ptr: *const T,
}

/// The basic Neighbour info returned by api
#[repr(C)]
pub struct Neighbour_api {
    /// id of neighbour
    pub id: usize,
    /// distance of data sent in request to this neighbour
    pub d: f32,
}

impl From<&Neighbour> for Neighbour_api {
    fn from(neighbour: &Neighbour) -> Self {
        Neighbour_api {
            id: neighbour.d_id,
            d: neighbour.distance,
        }
    }
}

/// The response to a neighbour search requests
#[repr(C)]
pub struct Neighbourhood_api {
    pub nbgh: i64,
    pub neighbours: *const Neighbour_api,
}

#[repr(C)]
pub struct Neighbour_api_parsearch_answer {
    /// The number of answers (o request), i.e size of both vectors nbgh and neighbours
    pub nb_answer: usize,
    /// for each request, we get a Neighbourhood_api
    pub neighbourhoods: *const Neighbourhood_api,
}

//===================================== Macros =====================================

#[allow(unused_macros)]
macro_rules! generate_insert(
    ($function_name:ident, $api_name:ty, $type_val:ty) => (
        #[no_mangle]
        pub unsafe extern "C" fn $function_name(hnsw_api : *mut $api_name, len:usize, data : *const $type_val, id : usize) {
            log::trace!("entering insert, type {:?} vec len is {:?}, id : {:?} ", stringify!($type_val), len, id);
            let data_v : Vec<$type_val>;
            unsafe {
                let slice = std::slice::from_raw_parts(data, len);
                data_v = Vec::from(slice);
                (*hnsw_api).opaque.insert_data(&data_v, id);
            }
        }
    )
);

#[allow(unused_macros)]
macro_rules! generate_parallel_insert(
    ($function_name:ident, $api_name:ty, $type_val:ty) => (
        #[no_mangle]
        pub unsafe extern "C" fn $function_name(hnsw_api : *mut $api_name, nb_vec: usize, vec_len : usize,
                        datas : *mut *const $type_val, ids : *const usize) {
            log::trace!("entering parallel_insert type {:?}  , vec len is {:?}, nb_vec : {:?}", stringify!($type_val), vec_len, nb_vec);
            let data_ids : Vec<usize>;
            let data_ptrs : Vec<*const $type_val>;
            unsafe {
                let slice = std::slice::from_raw_parts(ids, nb_vec);
                data_ids = Vec::from(slice);
                let slice_p = std::slice::from_raw_parts(datas, nb_vec);
                data_ptrs = Vec::from(slice_p);
            }
            let mut data_v = Vec::<Vec<$type_val>>::with_capacity(nb_vec);
            for i in 0..nb_vec {
                unsafe {
                    let slice = std::slice::from_raw_parts(data_ptrs[i], vec_len);
                    data_v.push(Vec::from(slice));
                }
            }
            let mut request : Vec<(&Vec<$type_val>, usize)> = Vec::with_capacity(nb_vec);
            for i in 0..nb_vec {
                request.push((&data_v[i], data_ids[i]));
            }
            unsafe { (*hnsw_api).opaque.parallel_insert_data(&request); };
        }
    )
);

#[allow(unused_macros)]
macro_rules! generate_search_neighbours(
    ($function_name:ident, $api_name:ty, $type_val:ty) => (
        #[no_mangle]
        pub unsafe extern "C" fn $function_name(hnsw_api : *const $api_name, len:usize, data : *const $type_val,
                                knbn : usize, ef_search : usize) ->  *const Neighbourhood_api {
            log::trace!("entering search_neighbours type {:?}, vec len is {:?}, id : {:?} ef_search {:?}", stringify!($type_val), len, knbn, ef_search);
            let data_v : Vec<$type_val>;
            let neighbours : Vec<Neighbour>;
            unsafe {
                let slice = std::slice::from_raw_parts(data, len);
                data_v = Vec::from(slice);
                neighbours =  (*hnsw_api).opaque.search_neighbours(&data_v, knbn, ef_search);
            }
            let neighbours_api : Vec<Neighbour_api> = neighbours.iter().map(|n| Neighbour_api::from(n)).collect();
            let nbgh_i = neighbours_api.len() as i64;
            let neighbours_ptr = neighbours_api.as_ptr();
            std::mem::forget(neighbours_api);
            let answer = Neighbourhood_api {
                    nbgh : nbgh_i,
                    neighbours : neighbours_ptr,
            };
            Box::into_raw(Box::new(answer))
        }
    )
);

#[allow(unused_macros)]
macro_rules! generate_parallel_search_neighbours(
    ($function_name:ident, $api_name:ty, $type_val:ty) => (
        #[no_mangle]
        pub unsafe extern "C" fn $function_name(hnsw_api : *const $api_name, nb_vec : usize, vec_len :i64,
                            data : *mut *const $type_val, knbn : usize, ef_search : usize) ->  *const Vec_api<Neighbourhood_api> {
            log::trace!("receiving parallel search request for type: {:?} with {:?} vectors", stringify!($type_val), nb_vec);
            let neighbours : Vec<Vec<Neighbour> >;
            let mut data_v = Vec::<Vec<$type_val>>::with_capacity(nb_vec);
            unsafe {
                let slice = std::slice::from_raw_parts(data, nb_vec);
                let ptr_list : Vec<*const $type_val> = Vec::from(slice);
                for i in 0..nb_vec {
                    let slice_i = std::slice::from_raw_parts(ptr_list[i], vec_len as usize);
                    data_v.push(Vec::from(slice_i));
                }
                neighbours =  (*hnsw_api).opaque.parallel_search_neighbours(&data_v, knbn, ef_search);
            }
            let mut neighbour_lists = Vec::<Neighbourhood_api>::with_capacity(nb_vec);
            for v in neighbours {
                let neighbours_api : Vec<Neighbour_api> = v.iter().map(|n| Neighbour_api::from(n)).collect();
                let nbgh = neighbours_api.len();
                let neighbours_api_ptr = neighbours_api.as_ptr();
                std::mem::forget(neighbours_api);
                neighbour_lists.push(Neighbourhood_api {
                    nbgh : nbgh as i64,
                    neighbours: neighbours_api_ptr,
                });
            }
            let neighbour_lists_ptr = neighbour_lists.as_ptr();
            std::mem::forget(neighbour_lists);
            let answer = Vec_api::<Neighbourhood_api> {
                len : nb_vec as i64,
                ptr : neighbour_lists_ptr,
            };
            Box::into_raw(Box::new(answer))
        }
    )
);

#[allow(unused_macros)]
macro_rules! generate_file_dump(
    ($function_name:ident, $api_name:ty, $type_val:ty) => (
    #[no_mangle]
        pub unsafe extern "C" fn $function_name(hnsw_api : *const $api_name, namelen : usize, filename :*const u8) -> i64 {
            let slice = unsafe { std::slice::from_raw_parts(filename, namelen) } ;
            let fstring  = String::from_utf8_lossy(slice).into_owned();
            let res =  unsafe { (*hnsw_api).opaque.file_dump(&fstring) } ;
            if res.is_ok() { 1 } else { -1 }
        }
    )
);

#[allow(unused_macros)]
macro_rules! generate_loadhnsw(
    ($function_name:ident, $api_name:ty, $type_val:ty, $type_dist : ty) => (
        #[no_mangle]
        pub unsafe extern "C" fn $function_name(flen : usize, name : *const u8)  -> *const $api_name {
            let  slice = unsafe { std::slice::from_raw_parts(name, flen)} ;
            let filename = String::from_utf8_lossy(slice).into_owned();
            let buffers = make_readers(&filename);
            let mut graph_in = buffers.0;
            let mut data_in = buffers.1;
            let hnsw_description = load_description(&mut graph_in).unwrap();
            let hnsw_loaded_res = load_hnsw::<$type_val, $type_dist>(&mut graph_in, &hnsw_description, &mut data_in);
            if let Ok(hnsw_loaded) = hnsw_loaded_res {
                let api = <$api_name>::new(Box::new(hnsw_loaded));
                return Box::into_raw(Box::new(api));
            }
            return ptr::null();
        }
     )
);

//===================================== Type Implementations =====================================

// f32
generate_loadhnsw!(
    load_hnswdump_f32_DistL1,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistL1
);
generate_loadhnsw!(
    load_hnswdump_f32_DistL2,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistL2
);
generate_loadhnsw!(
    load_hnswdump_f32_DistCosine,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistCosine
);
generate_loadhnsw!(
    load_hnswdump_f32_DistDot,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistDot
);
generate_loadhnsw!(
    load_hnswdump_f32_DistJensenShannon,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistJensenShannon
);
generate_loadhnsw!(
    load_hnswdump_f32_DistJeffreys,
    HnswApif32,
    f32,
    crate::core::index::hnsw_rs::DistJeffreys
);

// i32
generate_loadhnsw!(
    load_hnswdump_i32_DistL1,
    HnswApii32,
    i32,
    crate::core::index::hnsw_rs::DistL1
);
generate_loadhnsw!(
    load_hnswdump_i32_DistL2,
    HnswApii32,
    i32,
    crate::core::index::hnsw_rs::DistL2
);
generate_loadhnsw!(
    load_hnswdump_i32_DistHamming,
    HnswApii32,
    i32,
    crate::core::index::hnsw_rs::DistHamming
);

// u32
generate_loadhnsw!(
    load_hnswdump_u32_DistL1,
    HnswApiu32,
    u32,
    crate::core::index::hnsw_rs::DistL1
);
generate_loadhnsw!(
    load_hnswdump_u32_DistL2,
    HnswApiu32,
    u32,
    crate::core::index::hnsw_rs::DistL2
);
generate_loadhnsw!(
    load_hnswdump_u32_DistHamming,
    HnswApiu32,
    u32,
    crate::core::index::hnsw_rs::DistHamming
);
generate_loadhnsw!(
    load_hnswdump_u32_DistJaccard,
    HnswApiu32,
    u32,
    crate::core::index::hnsw_rs::DistJaccard
);

// u16
generate_loadhnsw!(
    load_hnswdump_u16_DistL1,
    HnswApiu16,
    u16,
    crate::core::index::hnsw_rs::DistL1
);
generate_loadhnsw!(
    load_hnswdump_u16_DistL2,
    HnswApiu16,
    u16,
    crate::core::index::hnsw_rs::DistL2
);
generate_loadhnsw!(
    load_hnswdump_u16_DistHamming,
    HnswApiu16,
    u16,
    crate::core::index::hnsw_rs::DistHamming
);
generate_loadhnsw!(
    load_hnswdump_u16_DistLevenshtein,
    HnswApiu16,
    u16,
    crate::core::index::hnsw_rs::DistLevenshtein
);

// u8
generate_loadhnsw!(
    load_hnswdump_u8_DistL1,
    HnswApiu8,
    u8,
    crate::core::index::hnsw_rs::DistL1
);
generate_loadhnsw!(
    load_hnswdump_u8_DistL2,
    HnswApiu8,
    u8,
    crate::core::index::hnsw_rs::DistL2
);
generate_loadhnsw!(
    load_hnswdump_u8_DistHamming,
    HnswApiu8,
    u8,
    crate::core::index::hnsw_rs::DistHamming
);
generate_loadhnsw!(
    load_hnswdump_u8_DistJaccard,
    HnswApiu8,
    u8,
    crate::core::index::hnsw_rs::DistJaccard
);

// NoData
generate_loadhnsw!(
    load_hnswdump_NoData_DistNoDist,
    HnswApiNodata,
    NoData,
    crate::core::index::hnsw_rs::NoDist
);

//===================================== Initialization functions =====================================

#[no_mangle]
pub unsafe extern "C" fn init_hnsw_f32(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
) -> *const HnswApif32 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice).into_owned();
    match dname.as_str() {
        "DistL1" => {
            let h = Hnsw::<f32, DistL1>::new(max_nb_conn, 10000, 16, ef_const, DistL1 {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistL2" => {
            let h = Hnsw::<f32, DistL2>::new(max_nb_conn, 10000, 16, ef_const, DistL2 {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistDot" => {
            let h = Hnsw::<f32, DistDot>::new(max_nb_conn, 10000, 16, ef_const, DistDot {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistHellinger" => {
            let h =
                Hnsw::<f32, DistHellinger>::new(max_nb_conn, 10000, 16, ef_const, DistHellinger {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistJeffreys" => {
            let h =
                Hnsw::<f32, DistJeffreys>::new(max_nb_conn, 10000, 16, ef_const, DistJeffreys {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistJensenShannon" => {
            let h = Hnsw::<f32, DistJensenShannon>::new(
                max_nb_conn,
                10000,
                16,
                ef_const,
                DistJensenShannon {},
            );
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        _ => ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn new_hnsw_f32(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
    max_elements: usize,
    max_layer: usize,
) -> *const HnswApif32 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    match dname.as_ref() {
        "DistL1" => {
            let h =
                Hnsw::<f32, DistL1>::new(max_nb_conn, max_elements, max_layer, ef_const, DistL1 {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistL2" => {
            let h =
                Hnsw::<f32, DistL2>::new(max_nb_conn, max_elements, max_layer, ef_const, DistL2 {});
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistDot" => {
            let h = Hnsw::<f32, DistDot>::new(
                max_nb_conn,
                max_elements,
                max_layer,
                ef_const,
                DistDot {},
            );
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistHellinger" => {
            let h = Hnsw::<f32, DistHellinger>::new(
                max_nb_conn,
                max_elements,
                max_layer,
                ef_const,
                DistHellinger {},
            );
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistJeffreys" => {
            let h = Hnsw::<f32, DistJeffreys>::new(
                max_nb_conn,
                max_elements,
                max_layer,
                ef_const,
                DistJeffreys {},
            );
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        "DistJensenShannon" => {
            let h = Hnsw::<f32, DistJensenShannon>::new(
                max_nb_conn,
                max_elements,
                max_layer,
                ef_const,
                DistJensenShannon {},
            );
            Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
        }
        _ => ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn drop_hnsw_f32(p: *const HnswApif32) {
    if !p.is_null() {
        let _raw = Box::from_raw(p as *mut HnswApif32);
    }
}

#[no_mangle]
pub unsafe extern "C" fn drop_hnsw_u16(p: *const HnswApiu16) {
    if !p.is_null() {
        let _raw = Box::from_raw(p as *mut HnswApiu16);
    }
}

#[no_mangle]
pub extern "C" fn init_hnsw_ptrdist_f32(
    max_nb_conn: usize,
    ef_const: usize,
    c_func: extern "C" fn(*const f32, *const f32, u64) -> f32,
) -> *const HnswApif32 {
    let c_dist = DistCFFI::<f32>::new(c_func);
    let h = Hnsw::<f32, DistCFFI<f32>>::new(max_nb_conn, 10000, 16, ef_const, c_dist);
    Box::into_raw(Box::new(HnswApif32::new(Box::new(h))))
}

// Implementations for other types
generate_insert!(insert_f32, HnswApif32, f32);
generate_parallel_insert!(parallel_insert_f32, HnswApif32, f32);
generate_search_neighbours!(search_neighbours_f32, HnswApif32, f32);
generate_parallel_search_neighbours!(parallel_search_neighbours_f32, HnswApif32, f32);
generate_file_dump!(file_dump_f32, HnswApif32, f32);

// i32
#[no_mangle]
pub unsafe extern "C" fn init_hnsw_i32(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
) -> *const HnswApii32 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    if dname == "DistL1" {
        let h = Hnsw::<i32, DistL1>::new(max_nb_conn, 10000, 16, ef_const, DistL1 {});
        return Box::into_raw(Box::new(HnswApii32::new(Box::new(h))));
    } else if dname == "DistL2" {
        let h = Hnsw::<i32, DistL2>::new(max_nb_conn, 10000, 16, ef_const, DistL2 {});
        return Box::into_raw(Box::new(HnswApii32::new(Box::new(h))));
    } else if dname == "DistHamming" {
        let h = Hnsw::<i32, DistHamming>::new(max_nb_conn, 10000, 16, ef_const, DistHamming {});
        return Box::into_raw(Box::new(HnswApii32::new(Box::new(h))));
    }
    ptr::null()
}

#[no_mangle]
pub extern "C" fn init_hnsw_ptrdist_i32(
    max_nb_conn: usize,
    ef_const: usize,
    c_func: extern "C" fn(*const i32, *const i32, u64) -> f32,
) -> *const HnswApii32 {
    let c_dist = DistCFFI::<i32>::new(c_func);
    let h = Hnsw::<i32, DistCFFI<i32>>::new(max_nb_conn, 10000, 16, ef_const, c_dist);
    Box::into_raw(Box::new(HnswApii32::new(Box::new(h))))
}

generate_insert!(insert_i32, HnswApii32, i32);
generate_parallel_insert!(parallel_insert_i32, HnswApii32, i32);
generate_search_neighbours!(search_neighbours_i32, HnswApii32, i32);
generate_parallel_search_neighbours!(parallel_search_neighbours_i32, HnswApii32, i32);
generate_file_dump!(file_dump_i32, HnswApii32, i32);

// u32
#[no_mangle]
pub unsafe extern "C" fn init_hnsw_u32(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
) -> *const HnswApiu32 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    if dname == "DistL1" {
        let h = Hnsw::<u32, DistL1>::new(max_nb_conn, 10000, 16, ef_const, DistL1 {});
        return Box::into_raw(Box::new(HnswApiu32::new(Box::new(h))));
    } else if dname == "DistL2" {
        let h = Hnsw::<u32, DistL2>::new(max_nb_conn, 10000, 16, ef_const, DistL2 {});
        return Box::into_raw(Box::new(HnswApiu32::new(Box::new(h))));
    } else if dname == "DistJaccard" {
        let h = Hnsw::<u32, DistJaccard>::new(max_nb_conn, 10000, 16, ef_const, DistJaccard {});
        return Box::into_raw(Box::new(HnswApiu32::new(Box::new(h))));
    } else if dname == "DistHamming" {
        let h = Hnsw::<u32, DistHamming>::new(max_nb_conn, 10000, 16, ef_const, DistHamming {});
        return Box::into_raw(Box::new(HnswApiu32::new(Box::new(h))));
    }
    ptr::null()
}

#[no_mangle]
pub extern "C" fn init_hnsw_ptrdist_u32(
    max_nb_conn: usize,
    ef_const: usize,
    c_func: extern "C" fn(*const u32, *const u32, u64) -> f32,
) -> *const HnswApiu32 {
    let c_dist = DistCFFI::<u32>::new(c_func);
    let h = Hnsw::<u32, DistCFFI<u32>>::new(max_nb_conn, 10000, 16, ef_const, c_dist);
    Box::into_raw(Box::new(HnswApiu32::new(Box::new(h))))
}

generate_insert!(insert_u32, HnswApiu32, u32);
generate_parallel_insert!(parallel_insert_u32, HnswApiu32, u32);
generate_search_neighbours!(search_neighbours_u32, HnswApiu32, u32);
generate_parallel_search_neighbours!(parallel_search_neighbours_u32, HnswApiu32, u32);
generate_file_dump!(file_dump_u32, HnswApiu32, u32);

// u16
#[no_mangle]
pub unsafe extern "C" fn init_hnsw_u16(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
) -> *const HnswApiu16 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    if dname == "DistL1" {
        let h = Hnsw::<u16, DistL1>::new(max_nb_conn, 10000, 16, ef_const, DistL1 {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistL2" {
        let h = Hnsw::<u16, DistL2>::new(max_nb_conn, 10000, 16, ef_const, DistL2 {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistHamming" {
        let h = Hnsw::<u16, DistHamming>::new(max_nb_conn, 10000, 16, ef_const, DistHamming {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistJaccard" {
        let h = Hnsw::<u16, DistJaccard>::new(max_nb_conn, 10000, 16, ef_const, DistJaccard {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistLevenshtein" {
        let h =
            Hnsw::<u16, DistLevenshtein>::new(max_nb_conn, 10000, 16, ef_const, DistLevenshtein {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    }
    ptr::null()
}

#[no_mangle]
pub unsafe extern "C" fn new_hnsw_u16(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
    max_elements: usize,
    max_layer: usize,
) -> *const HnswApiu16 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    if dname == "DistL1" {
        let h = Hnsw::<u16, DistL1>::new(max_nb_conn, max_elements, max_layer, ef_const, DistL1 {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistL2" {
        let h = Hnsw::<u16, DistL2>::new(max_nb_conn, max_elements, max_layer, ef_const, DistL2 {});
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistHamming" {
        let h = Hnsw::<u16, DistHamming>::new(
            max_nb_conn,
            max_elements,
            max_layer,
            ef_const,
            DistHamming {},
        );
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistJaccard" {
        let h = Hnsw::<u16, DistJaccard>::new(
            max_nb_conn,
            max_elements,
            max_layer,
            ef_const,
            DistJaccard {},
        );
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    } else if dname == "DistLevenshtein" {
        let h = Hnsw::<u16, DistLevenshtein>::new(
            max_nb_conn,
            max_elements,
            max_layer,
            ef_const,
            DistLevenshtein {},
        );
        return Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))));
    }
    ptr::null()
}

#[no_mangle]
pub extern "C" fn init_hnsw_ptrdist_u16(
    max_nb_conn: usize,
    ef_const: usize,
    c_func: extern "C" fn(*const u16, *const u16, u64) -> f32,
) -> *const HnswApiu16 {
    let c_dist = DistCFFI::<u16>::new(c_func);
    let h = Hnsw::<u16, DistCFFI<u16>>::new(max_nb_conn, 10000, 16, ef_const, c_dist);
    Box::into_raw(Box::new(HnswApiu16::new(Box::new(h))))
}

generate_insert!(insert_u16, HnswApiu16, u16);
generate_parallel_insert!(parallel_insert_u16, HnswApiu16, u16);
generate_search_neighbours!(search_neighbours_u16, HnswApiu16, u16);
generate_parallel_search_neighbours!(parallel_search_neighbours_u16, HnswApiu16, u16);
generate_file_dump!(file_dump_u16, HnswApiu16, u16);

// u8
#[no_mangle]
pub unsafe extern "C" fn init_hnsw_u8(
    max_nb_conn: usize,
    ef_const: usize,
    namelen: usize,
    cdistname: *const u8,
) -> *const HnswApiu8 {
    let slice = unsafe { std::slice::from_raw_parts(cdistname, namelen) };
    let dname = String::from_utf8_lossy(slice);
    if dname == "DistL1" {
        let h = Hnsw::<u8, DistL1>::new(max_nb_conn, 10000, 16, ef_const, DistL1 {});
        return Box::into_raw(Box::new(HnswApiu8::new(Box::new(h))));
    } else if dname == "DistL2" {
        let h = Hnsw::<u8, DistL2>::new(max_nb_conn, 10000, 16, ef_const, DistL2 {});
        return Box::into_raw(Box::new(HnswApiu8::new(Box::new(h))));
    } else if dname == "DistHamming" {
        let h = Hnsw::<u8, DistHamming>::new(max_nb_conn, 10000, 16, ef_const, DistHamming {});
        return Box::into_raw(Box::new(HnswApiu8::new(Box::new(h))));
    } else if dname == "DistJaccard" {
        let h = Hnsw::<u8, DistJaccard>::new(max_nb_conn, 10000, 16, ef_const, DistJaccard {});
        return Box::into_raw(Box::new(HnswApiu8::new(Box::new(h))));
    }
    ptr::null()
}

#[no_mangle]
pub extern "C" fn init_hnsw_ptrdist_u8(
    max_nb_conn: usize,
    ef_const: usize,
    c_func: extern "C" fn(*const u8, *const u8, u64) -> f32,
) -> *const HnswApiu8 {
    let c_dist = DistCFFI::<u8>::new(c_func);
    let h = Hnsw::<u8, DistCFFI<u8>>::new(max_nb_conn, 10000, 16, ef_const, c_dist);
    Box::into_raw(Box::new(HnswApiu8::new(Box::new(h))))
}

generate_insert!(insert_u8, HnswApiu8, u8);
generate_parallel_insert!(parallel_insert_u8, HnswApiu8, u8);
generate_search_neighbours!(search_neighbours_u8, HnswApiu8, u8);
generate_parallel_search_neighbours!(parallel_search_neighbours_u8, HnswApiu8, u8);
generate_file_dump!(file_dump_u8, HnswApiu8, u8);

//=========================== dump restore functions ===========================

#[repr(C)]
pub struct DescriptionFFI {
    pub dumpmode: u8,
    pub max_nb_connection: u8,
    pub nb_layer: u8,
    pub ef: usize,
    pub nb_point: usize,
    pub data_dimension: usize,
    pub distname_len: usize,
    pub distname: *const u8,
    pub t_name_len: usize,
    pub t_name: *const u8,
}

impl Default for DescriptionFFI {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptionFFI {
    pub fn new() -> Self {
        DescriptionFFI {
            dumpmode: 0,
            max_nb_connection: 0,
            nb_layer: 0,
            ef: 0,
            nb_point: 0,
            data_dimension: 0,
            distname_len: 0,
            distname: ptr::null(),
            t_name_len: 0,
            t_name: ptr::null(),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn load_hnsw_description(
    flen: usize,
    name: *const u8,
) -> *const DescriptionFFI {
    let slice = unsafe { std::slice::from_raw_parts(name, flen) };
    let filename = String::from_utf8_lossy(slice).into_owned();
    let fpath = PathBuf::from(filename);
    let fileres = OpenOptions::new().read(true).open(&fpath);
    let mut ffi_description = DescriptionFFI::new();
    match fileres {
        Ok(file) => {
            let mut bufr = BufReader::with_capacity(10_000_000, file);
            let res = load_description(&mut bufr);
            if let Ok(description) = res {
                let distname = description.distname.clone();
                let distname_ptr = distname.as_ptr();
                let distname_len = distname.len();
                std::mem::forget(distname);

                let t_name = description.t_name.clone();
                let t_name_ptr = t_name.as_ptr();
                let t_name_len = t_name.len();
                std::mem::forget(t_name);

                ffi_description.dumpmode = 1;
                ffi_description.max_nb_connection = description.max_nb_connection;
                ffi_description.nb_layer = description.nb_layer;
                ffi_description.ef = description.ef;
                ffi_description.data_dimension = description.dimension;
                ffi_description.distname_len = distname_len;
                ffi_description.distname = distname_ptr;
                ffi_description.t_name_len = t_name_len;
                ffi_description.t_name = t_name_ptr;
                return Box::into_raw(Box::new(ffi_description));
            }
            ptr::null()
        }
        Err(_) => ptr::null(),
    }
}

pub fn make_readers(basename: &String) -> (BufReader<std::fs::File>, BufReader<std::fs::File>) {
    let mut graphfname = basename.clone();
    graphfname.push_str(".hnsw.graph");
    let graphpath = PathBuf::from(graphfname);
    let graphfileres = OpenOptions::new().read(true).open(&graphpath);
    if graphfileres.is_err() {
        std::panic::panic_any(format!(
            "make_readers : could not open file {:?}",
            graphpath.as_os_str()
        ));
    }
    let graphfile = graphfileres.unwrap();

    let mut datafname = basename.clone();
    datafname.push_str(".hnsw.data");
    let datapath = PathBuf::from(datafname);
    let datafileres = OpenOptions::new().read(true).open(&datapath);
    if datafileres.is_err() {
        std::panic::panic_any(format!(
            "make_readers: could not open file {:?}",
            datapath.as_os_str()
        ));
    }
    let datafile = datafileres.unwrap();
    (BufReader::new(graphfile), BufReader::new(datafile))
}

//============ log initialization ============//

#[no_mangle]
pub extern "C" fn init_rust_log() {
    let _res = env_logger::Builder::from_default_env().try_init();
}
