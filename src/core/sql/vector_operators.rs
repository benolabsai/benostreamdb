use datafusion::error::Result;
use datafusion::prelude::SessionContext;

/// Mapping structure for pgvector operators to their corresponding distance UDFs
#[derive(Debug, Clone)]
pub struct VectorOperatorMapping {
    pub operator: &'static str,
    pub udf_name: &'static str,
    pub description: &'static str,
}

/// All pgvector-compatible operators with their UDF mappings
pub const VECTOR_OPERATORS: &[VectorOperatorMapping] = &[
    VectorOperatorMapping {
        operator: "<->",
        udf_name: "dist_l2",
        description: "L2 distance (Euclidean)",
    },
    VectorOperatorMapping {
        operator: "<=>",
        udf_name: "dist_cosine",
        description: "Cosine distance",
    },
    VectorOperatorMapping {
        operator: "<#>",
        udf_name: "dist_ip",
        description: "Inner product distance (negative dot product)",
    },
    VectorOperatorMapping {
        operator: "<+>",
        udf_name: "dist_l1",
        description: "L1 distance (Manhattan)",
    },
    VectorOperatorMapping {
        operator: "<~>",
        udf_name: "dist_hamming",
        description: "Hamming distance",
    },
    VectorOperatorMapping {
        operator: "<%>",
        udf_name: "dist_jaccard",
        description: "Jaccard distance",
    },
];

/// Register all pgvector-compatible vector operators in the DataFusion session
/// 
/// This function maps pgvector operators (<->, <=>, <#>, <+>, <~>, <%>) to their
/// corresponding distance UDFs (dist_l2, dist_cosine, dist_ip, dist_l1, dist_hamming, dist_jaccard).
/// 
/// # Arguments
/// * `session` - The DataFusion SessionContext to register operators in
/// 
/// # Returns
/// * `Result<()>` - Ok if all operators were registered successfully
/// 
/// # Example
/// ```no_run
/// use datafusion::prelude::SessionContext;
/// use hyperstreamdb::core::sql::vector_operators::register_vector_operators;
/// 
/// let mut ctx = SessionContext::new();
/// register_vector_operators(&mut ctx).unwrap();
/// ```
pub fn register_vector_operators(session: &mut SessionContext) -> Result<()> {
    // Note: DataFusion 52+ doesn't have a direct API for registering custom operators
    // as syntax extensions. The operators are handled at the parser level.
    // 
    // The actual operator parsing happens in the SQL parser (PostgreSQL dialect),
    // and the optimizer (VectorSearchOptimizerRule) already handles these operators
    // by checking the operator string representation.
    // 
    // This function serves as a documentation point and validation that all required
    // UDFs are registered. In the future, if DataFusion adds custom operator registration,
    // this is where we would register them.
    
    // Verify that all required UDFs are registered
    for mapping in VECTOR_OPERATORS {
        // Check if the UDF exists in the session
        let state = session.state();
        let udf_exists = state.scalar_functions().get(mapping.udf_name).is_some();
        
        if !udf_exists {
            return Err(datafusion::error::DataFusionError::Plan(
                format!(
                    "Vector operator {} requires UDF {} to be registered, but it was not found",
                    mapping.operator, mapping.udf_name
                )
            ));
        }
    }
    
    Ok(())
}

/// Get the UDF name for a given vector operator
/// 
/// # Arguments
/// * `operator` - The operator string (e.g., "<->", "<=>")
/// 
/// # Returns
/// * `Option<&'static str>` - The corresponding UDF name, or None if not found
pub fn get_udf_for_operator(operator: &str) -> Option<&'static str> {
    VECTOR_OPERATORS
        .iter()
        .find(|m| m.operator == operator)
        .map(|m| m.udf_name)
}

/// Get the operator string for a given UDF name
/// 
/// # Arguments
/// * `udf_name` - The UDF name (e.g., "dist_l2", "dist_cosine")
/// 
/// # Returns
/// * `Option<&'static str>` - The corresponding operator string, or None if not found
pub fn get_operator_for_udf(udf_name: &str) -> Option<&'static str> {
    VECTOR_OPERATORS
        .iter()
        .find(|m| m.udf_name == udf_name)
        .map(|m| m.operator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;
    use crate::core::sql::vector_udf;

    #[test]
    fn test_operator_mapping_completeness() {
        // Verify all six operators are defined
        assert_eq!(VECTOR_OPERATORS.len(), 6);
        
        // Verify all expected operators are present
        let operators: Vec<&str> = VECTOR_OPERATORS.iter().map(|m| m.operator).collect();
        assert!(operators.contains(&"<->"));
        assert!(operators.contains(&"<=>"));
        assert!(operators.contains(&"<#>"));
        assert!(operators.contains(&"<+>"));
        assert!(operators.contains(&"<~>"));
        assert!(operators.contains(&"<%>"));
        
        // Verify all expected UDFs are mapped
        let udfs: Vec<&str> = VECTOR_OPERATORS.iter().map(|m| m.udf_name).collect();
        assert!(udfs.contains(&"dist_l2"));
        assert!(udfs.contains(&"dist_cosine"));
        assert!(udfs.contains(&"dist_ip"));
        assert!(udfs.contains(&"dist_l1"));
        assert!(udfs.contains(&"dist_hamming"));
        assert!(udfs.contains(&"dist_jaccard"));
    }

    #[test]
    fn test_get_udf_for_operator() {
        assert_eq!(get_udf_for_operator("<->"), Some("dist_l2"));
        assert_eq!(get_udf_for_operator("<=>"), Some("dist_cosine"));
        assert_eq!(get_udf_for_operator("<#>"), Some("dist_ip"));
        assert_eq!(get_udf_for_operator("<+>"), Some("dist_l1"));
        assert_eq!(get_udf_for_operator("<~>"), Some("dist_hamming"));
        assert_eq!(get_udf_for_operator("<%>"), Some("dist_jaccard"));
        assert_eq!(get_udf_for_operator("<unknown>"), None);
    }

    #[test]
    fn test_get_operator_for_udf() {
        assert_eq!(get_operator_for_udf("dist_l2"), Some("<->"));
        assert_eq!(get_operator_for_udf("dist_cosine"), Some("<=>"));
        assert_eq!(get_operator_for_udf("dist_ip"), Some("<#>"));
        assert_eq!(get_operator_for_udf("dist_l1"), Some("<+>"));
        assert_eq!(get_operator_for_udf("dist_hamming"), Some("<~>"));
        assert_eq!(get_operator_for_udf("dist_jaccard"), Some("<%>"));
        assert_eq!(get_operator_for_udf("unknown_udf"), None);
    }

    #[test]
    fn test_register_vector_operators_success() {
        let mut ctx = SessionContext::new();
        
        // Register all vector UDFs first
        for udf in vector_udf::all_vector_udfs() {
            ctx.register_udf(udf);
        }
        
        // Now register operators should succeed
        let result = register_vector_operators(&mut ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_register_vector_operators_missing_udf() {
        let mut ctx = SessionContext::new();
        
        // Don't register UDFs - should fail
        let result = register_vector_operators(&mut ctx);
        assert!(result.is_err());
        
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("requires UDF"));
    }

    // Unit tests for edge cases - Task 15.1
    // Test each operator individually
    
    #[test]
    fn test_l2_operator_mapping() {
        assert_eq!(get_udf_for_operator("<->"), Some("dist_l2"));
        assert_eq!(get_operator_for_udf("dist_l2"), Some("<->"));
        
        // Verify the mapping is in VECTOR_OPERATORS
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<->");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_l2");
        assert!(m.description.contains("L2") || m.description.contains("Euclidean"));
    }
    
    #[test]
    fn test_cosine_operator_mapping() {
        assert_eq!(get_udf_for_operator("<=>"), Some("dist_cosine"));
        assert_eq!(get_operator_for_udf("dist_cosine"), Some("<=>"));
        
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<=>");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_cosine");
        assert!(m.description.contains("Cosine"));
    }
    
    #[test]
    fn test_inner_product_operator_mapping() {
        assert_eq!(get_udf_for_operator("<#>"), Some("dist_ip"));
        assert_eq!(get_operator_for_udf("dist_ip"), Some("<#>"));
        
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<#>");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_ip");
        assert!(m.description.contains("Inner product") || m.description.contains("dot product"));
    }
    
    #[test]
    fn test_l1_operator_mapping() {
        assert_eq!(get_udf_for_operator("<+>"), Some("dist_l1"));
        assert_eq!(get_operator_for_udf("dist_l1"), Some("<+>"));
        
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<+>");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_l1");
        assert!(m.description.contains("L1") || m.description.contains("Manhattan"));
    }
    
    #[test]
    fn test_hamming_operator_mapping() {
        assert_eq!(get_udf_for_operator("<~>"), Some("dist_hamming"));
        assert_eq!(get_operator_for_udf("dist_hamming"), Some("<~>"));
        
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<~>");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_hamming");
        assert!(m.description.contains("Hamming"));
    }
    
    #[test]
    fn test_jaccard_operator_mapping() {
        assert_eq!(get_udf_for_operator("<%>"), Some("dist_jaccard"));
        assert_eq!(get_operator_for_udf("dist_jaccard"), Some("<%>"));
        
        let mapping = VECTOR_OPERATORS.iter().find(|m| m.operator == "<%>");
        assert!(mapping.is_some());
        let m = mapping.unwrap();
        assert_eq!(m.udf_name, "dist_jaccard");
        assert!(m.description.contains("Jaccard"));
    }
    
    #[test]
    fn test_operator_uniqueness() {
        // Verify all operators are unique
        let mut operators = std::collections::HashSet::new();
        for mapping in VECTOR_OPERATORS {
            assert!(operators.insert(mapping.operator), 
                "Duplicate operator found: {}", mapping.operator);
        }
        assert_eq!(operators.len(), 6);
    }
    
    #[test]
    fn test_udf_uniqueness() {
        // Verify all UDF names are unique
        let mut udfs = std::collections::HashSet::new();
        for mapping in VECTOR_OPERATORS {
            assert!(udfs.insert(mapping.udf_name), 
                "Duplicate UDF found: {}", mapping.udf_name);
        }
        assert_eq!(udfs.len(), 6);
    }
    
    #[test]
    fn test_operator_not_found() {
        // Test invalid operators return None
        assert_eq!(get_udf_for_operator("<<>>"), None);
        assert_eq!(get_udf_for_operator(""), None);
        assert_eq!(get_udf_for_operator("invalid"), None);
        assert_eq!(get_udf_for_operator("<"), None);
        assert_eq!(get_udf_for_operator(">"), None);
    }
    
    #[test]
    fn test_udf_not_found() {
        // Test invalid UDF names return None
        assert_eq!(get_operator_for_udf("dist_invalid"), None);
        assert_eq!(get_operator_for_udf(""), None);
        assert_eq!(get_operator_for_udf("distance"), None);
        assert_eq!(get_operator_for_udf("dist_"), None);
    }
    
    #[test]
    fn test_operator_case_sensitivity() {
        // Operators should be case-sensitive
        assert_eq!(get_udf_for_operator("<->"), Some("dist_l2"));
        assert_eq!(get_udf_for_operator("<-<"), None); // Different operator
    }
    
    #[test]
    fn test_all_operators_have_descriptions() {
        // Verify all operators have non-empty descriptions
        for mapping in VECTOR_OPERATORS {
            assert!(!mapping.description.is_empty(), 
                "Operator {} has empty description", mapping.operator);
            assert!(mapping.description.len() > 5, 
                "Operator {} has too short description", mapping.operator);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use datafusion::prelude::SessionContext;
    use crate::core::sql::vector_udf;

    // Feature: pgvector-sql-support, Property 1: Operator Parsing Correctness
    // **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
    // 
    // Property: For any valid SQL query containing a pgvector distance operator 
    // (<->, <=>, <#>, <+>, <~>, <%>), the parser should correctly map the operator 
    // to its corresponding distance function (L2, Cosine, Inner Product, L1, Hamming, 
    // Jaccard respectively).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        
        #[test]
        fn test_operator_to_udf_mapping_correctness(
            operator in prop::sample::select(vec!["<->", "<=>", "<#>", "<+>", "<~>", "<%>"])
        ) {
            // Verify that each operator has a corresponding UDF mapping
            let udf_name = get_udf_for_operator(&operator);
            prop_assert!(udf_name.is_some(), "Operator {} should have a UDF mapping", operator);
            
            // Verify the mapping is bidirectional
            let udf = udf_name.unwrap();
            let reverse_op = get_operator_for_udf(udf);
            prop_assert_eq!(reverse_op, Some(operator), 
                "UDF {} should map back to operator {}", udf, operator);
            
            // Verify the UDF name matches expected pattern
            prop_assert!(udf.starts_with("dist_"), 
                "UDF name {} should start with 'dist_'", udf);
        }
        
        #[test]
        fn test_all_operators_have_unique_udfs(
            idx1 in 0usize..6,
            idx2 in 0usize..6
        ) {
            let op1 = &VECTOR_OPERATORS[idx1];
            let op2 = &VECTOR_OPERATORS[idx2];
            
            if idx1 != idx2 {
                // Different operators should map to different UDFs
                prop_assert_ne!(op1.udf_name, op2.udf_name,
                    "Operators {} and {} should map to different UDFs", 
                    op1.operator, op2.operator);
                
                // Different operators should have different operator strings
                prop_assert_ne!(op1.operator, op2.operator,
                    "Operators at indices {} and {} should be different", idx1, idx2);
            }
        }
        
        #[test]
        fn test_operator_registration_with_random_subset(
            // Generate a random subset of operators to test partial registration
            include_l2 in any::<bool>(),
            include_cosine in any::<bool>(),
            include_ip in any::<bool>(),
            include_l1 in any::<bool>(),
            include_hamming in any::<bool>(),
            include_jaccard in any::<bool>()
        ) {
            let mut ctx = SessionContext::new();
            
            // Register a subset of UDFs based on random booleans
            let mut registered_count = 0;
            for udf in vector_udf::all_vector_udfs() {
                let name = udf.name().to_string();
                let should_register = match name.as_str() {
                    "dist_l2" => include_l2,
                    "dist_cosine" => include_cosine,
                    "dist_ip" => include_ip,
                    "dist_l1" => include_l1,
                    "dist_hamming" => include_hamming,
                    "dist_jaccard" => include_jaccard,
                    _ => true, // Register other UDFs
                };
                
                if should_register {
                    if matches!(name.as_str(), "dist_l2" | "dist_cosine" | "dist_ip" | "dist_l1" | "dist_hamming" | "dist_jaccard") {
                        registered_count += 1;
                    }
                    ctx.register_udf(udf);
                }
            }
            
            // Registration should succeed only if all 6 distance UDFs are registered
            let result = register_vector_operators(&mut ctx);
            if registered_count == 6 {
                prop_assert!(result.is_ok(), 
                    "Registration should succeed when all 6 UDFs are present");
            } else {
                prop_assert!(result.is_err(), 
                    "Registration should fail when only {} of 6 UDFs are present", 
                    registered_count);
            }
        }
    }
}
