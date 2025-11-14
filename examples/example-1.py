def calculate_total(items, tax_rate=0.08, discount=None):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    if discount:
        subtotal *= (1 - discount)
    return subtotal * (1 + tax_rate)