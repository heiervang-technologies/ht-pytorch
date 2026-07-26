#[cfg(test)]
mod unit_tests {
    use crate::allocator::round_up_power_of_two;

    #[test]
    fn allocation_buckets_are_checked_and_have_a_minimum_size() {
        assert_eq!(round_up_power_of_two(0), Some(256));
        assert_eq!(round_up_power_of_two(255), Some(256));
        assert_eq!(round_up_power_of_two(257), Some(512));
        assert_eq!(round_up_power_of_two(usize::MAX), None);
    }
}
