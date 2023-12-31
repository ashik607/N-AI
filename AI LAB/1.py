def tower_of_hanoi(n, from_rod, aux_rod, to_rod):
    if n == 1:
        print(f"Move disk 1 from rod {from_rod} to rod {to_rod}")
        return
    tower_of_hanoi(n - 1, from_rod, to_rod, aux_rod)
    print(f"Move disk {n} from rod {from_rod} to rod {to_rod}")
    tower_of_hanoi(n - 1, aux_rod, from_rod, to_rod)

if __name__ == "__main__":
    n = int(input("Enter number of disks: "))  # Number of disks
    tower_of_hanoi(n, 'A', 'B', 'C')  # A, B, C are names of rods



# Algorithm: Tower of Hanoi

# Input: Number of disks n, source rod from_rod, auxiliary rod aux_rod, target rod to_rod.

# Base Case:

# If n equals 1:
# Print the move: "Move disk from {from_rod} to {to_rod}".
# Return.
# Recursive Steps:

# Move n-1 disks from from_rod to aux_rod using to_rod as auxiliary.
# Recur: tower_of_hanoi(n - 1, from_rod, to_rod, aux_rod).
# Print the move: "Move disk from {from_rod} to {to_rod}".
# Move the n-1 disks from aux_rod to to_rod using from_rod as auxiliary.
# Recur: tower_of_hanoi(n - 1, aux_rod, from_rod, to_rod).
# Output: The sequence of moves to solve the Tower of Hanoi problem for n disks.