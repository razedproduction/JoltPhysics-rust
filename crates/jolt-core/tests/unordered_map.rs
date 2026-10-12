//! Port of UnitTests/Core/UnorderedMapTest.cpp

use jolt_core::UnorderedMap;

#[test]
fn test_unordered_map() {
    let mut map: UnorderedMap<i32, i32> = UnorderedMap::new();
    map.reserve(10);

    // Insert some entries
    assert_eq!(map.insert((1, 2)).0.0, 1);
    assert!(map.insert((3, 4)).1);
    assert!(!map.insert((3, 5)).1);
    assert_eq!(map.len(), 2);
    assert_eq!(map.find(&1).unwrap().1, 2);
    assert_eq!(map.find(&3).unwrap().1, 4);
    assert!(map.find(&5).is_none());

    // Use operator []
    *map.get_or_insert_default(5) = 6;
    assert_eq!(map.len(), 3);
    assert_eq!(map.find(&5).unwrap().1, 6);
    *map.get_or_insert_default(5) = 7;
    assert_eq!(map.len(), 3);
    assert_eq!(map.find(&5).unwrap().1, 7);

    // Validate all elements are visited by a visitor
    let mut count = 0;
    let mut visited = [false; 10];
    for i in map.iter() {
        visited[i.0 as usize] = true;
        count += 1;
    }
    assert_eq!(count, 3);
    assert!(visited[1]);
    assert!(visited[3]);
    assert!(visited[5]);
    for i in map.iter_mut() {
        visited[i.0 as usize] = false;
        count -= 1;
    }
    assert_eq!(count, 0);
    assert!(!visited[1]);
    assert!(!visited[3]);
    assert!(!visited[5]);

    // Copy the map
    let mut map2: UnorderedMap<i32, i32> = UnorderedMap::new();
    assert!(map2.is_empty());
    map2 = map.clone();
    assert_eq!(map2.find(&1).unwrap().1, 2);
    assert_eq!(map2.find(&3).unwrap().1, 4);
    assert_eq!(map2.find(&5).unwrap().1, 7);
    assert!(map2.find(&7).is_none());

    // Try emplace
    map.try_emplace(7, 8);
    assert_eq!(map.len(), 4);
    assert_eq!(map.find(&7).unwrap().1, 8);

    // Swap
    let mut map3: UnorderedMap<i32, i32> = UnorderedMap::new();
    map3.swap(&mut map);
    assert_eq!(map3.find(&1).unwrap().1, 2);
    assert_eq!(map3.find(&3).unwrap().1, 4);
    assert_eq!(map3.find(&5).unwrap().1, 7);
    assert_eq!(map3.find(&7).unwrap().1, 8);
    assert!(map3.find(&9).is_none());
    assert!(map.is_empty());

    // Move construct
    let map4 = std::mem::take(&mut map3);
    assert_eq!(map4.find(&1).unwrap().1, 2);
    assert_eq!(map4.find(&3).unwrap().1, 4);
    assert_eq!(map4.find(&5).unwrap().1, 7);
    assert_eq!(map4.find(&7).unwrap().1, 8);
    assert!(map4.find(&9).is_none());
    assert!(map3.is_empty());
}

#[test]
fn test_unordered_map_grow() {
    let mut map: UnorderedMap<i32, i32> = UnorderedMap::new();
    for i in 0..10000 {
        assert!(map.try_emplace(i, !i).1);
    }

    assert_eq!(map.len(), 10000);

    for i in 0..10000 {
        assert_eq!(map.find(&i).unwrap().1, !i);
    }

    assert!(map.find(&10001).is_none());

    for i in 0..5000 {
        assert_eq!(map.erase(&i), 1);
    }

    assert_eq!(map.len(), 5000);

    for i in 0..5000 {
        assert!(map.find(&i).is_none());
    }

    for i in 5000..10000 {
        assert_eq!(map.find(&i).unwrap().1, !i);
    }

    assert!(map.find(&10001).is_none());

    for i in 0..5000 {
        assert!(map.try_emplace(i, i + 1).1);
    }

    assert!(!map.try_emplace(0, 0).1);

    assert_eq!(map.len(), 10000);

    for i in 0..5000 {
        assert_eq!(map.find(&i).unwrap().1, i + 1);
    }

    for i in 5000..10000 {
        assert_eq!(map.find(&i).unwrap().1, !i);
    }

    assert!(map.find(&10001).is_none());
}
