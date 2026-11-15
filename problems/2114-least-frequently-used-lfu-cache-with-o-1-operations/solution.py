class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.freq = 1
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_head(self, node):
        """Add node right after head (Most Recently Used in this freq list)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove_node(self, node):
        """Remove an arbitrary node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def pop_tail(self):
        """Remove and return the least recently used node (just before tail)."""
        if self.size == 0:
            return None
        lru_node = self.tail.prev
        self.remove_node(lru_node)
        return lru_node

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.min_freq = 0
        self.key_map = {}  # key -> Node
        self.freq_map = {} # freq -> DoublyLinkedList

    def _update_freq(self, node):
        """Increment node's frequency and update data structures."""
        freq = node.freq
        self.freq_map[freq].remove_node(node)
        
        # If the min_freq list becomes empty, increment min_freq
        if freq == self.min_freq and self.freq_map[freq].size == 0:
            self.min_freq += 1
            
        node.freq += 1
        new_freq = node.freq
        if new_freq not in self.freq_map:
            self.freq_map[new_freq] = DoublyLinkedList()
        self.freq_map[new_freq].add_to_head(node)

    def get(self, key: int) -> int:
        if key not in self.key_map:
            return -1
        node = self.key_map[key]
        self._update_freq(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
            
        if key in self.key_map:
            node = self.key_map[key]
            node.val = value
            self._update_freq(node)
            return
            
        # If cache is full, evict the LFU-LRU item
        if self.size >= self.capacity:
            lru_list = self.freq_map[self.min_freq]
            evicted = lru_list.pop_tail()
            if evicted:
                del self.key_map[evicted.key]
                self.size -= 1
                
        # Insert new key
        new_node = Node(key, value)
        self.key_map[key] = new_node
        self.min_freq = 1
        if 1 not in self.freq_map:
            self.freq_map[1] = DoublyLinkedList()
        self.freq_map[1].add_to_head(new_node)
        self.size += 1