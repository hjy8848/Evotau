# Activation-smoke task candidates — UNREVIEWED

This is a human review worksheet generated from the pinned τ-bench train split. It does not freeze tasks and cannot be consumed by the runner. All candidates, including Task 73, still require explicit researcher eligibility review. Test tasks were not loaded.

Pinned data: `sierra-research/tau2-bench` `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, package `1.0.1`. Validation remains Task 93.

For each task, mark panel inclusion and all four eligibility checks in the JSON worksheet; provide reviewer identity, UTC timestamp, rationale, and exclusion reason where needed. After selecting exactly ten, record final pairwise diversity in the existing task-semantic-review format.

## Candidate 1: Task 16
- Task SHA-256: `a46a7848d29777c7d6bb3beadea6d3cafa0da12158830a6a0c685f90fb08ca55`
- Workflow hypothesis: pending-order cancellation, delivered-order return/refund
- Reference tools: find_user_id_by_name_zip, get_user_details, get_order_details, calculate, cancel_pending_order, return_delivered_order_items
- Reference write actions: `[{"tool": "cancel_pending_order", "arguments": {"order_id": "#W5199551", "reason": "no longer needed"}}, {"tool": "cancel_pending_order", "arguments": {"order_id": "#W8665881", "reason": "no longer needed"}}, {"tool": "return_delivered_order_items", "arguments": {"order_id": "#W9389413", "item_ids": ["2554056026"], "payment_method_id": "paypal_5364164"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 86-92: pending status and allowed cancellation reason.; Retail policy lines 116-126: delivered status, item list, refund method, confirmation.
- User scenario: You want to cancel all pending orders (since they are no longer needed) and return the watch you have received (but nothing else), and you want to know the total amount you can get back.
- Known facts: You are Fatima Johnson in zipcode 78712.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 2: Task 22
- Task SHA-256: `968d73ce2d67f42ca10f1fd2b4368ac05db7bdf842bfcc26ca1afe1fec221b7f`
- Workflow hypothesis: customer default-address update, pending-order address change
- Reference tools: find_user_id_by_name_zip, modify_user_address, get_order_details, modify_pending_order_address
- Reference write actions: `[{"tool": "modify_user_address", "arguments": {"user_id": "ethan_garcia_1261", "address1": "101 Highway", "address2": "", "city": "New York", "state": "NY", "country": "USA", "zip": "10001"}}, {"tool": "modify_pending_order_address", "arguments": {"order_id": "#W9911714", "address1": "101 Highway", "address2": "", "city": "New York", "state": "NY", "country": "USA", "zip": "10001"}}, {"tool": "modify_user_address", "arguments": {"user_id": "ethan_garcia_1261", "address1": "667 Highland Drive", "address2": "Suite 865", "city": "Denver", "state": "CO", "country": "USA", "zip": "80280"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.
- User scenario: You want to change your user address and all possible order addresses to be 101 Highway, New York, 10001 (including the orders). Once the agent has confirmed the changes for the default address AND the order addresses, then you regret it and want to change the user address back to the original address.
- Known facts: You are Ethan Garcia, and you live in Denver, 80280.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 3: Task 29
- Task SHA-256: `c12c34a735c9ac2d54991ef3c0f90e85c301a16e4731511266712ccd88d9125d`
- Workflow hypothesis: delivered-order exchange
- Reference tools: find_user_id_by_name_zip, get_user_details, get_order_details, get_product_details, exchange_delivered_order_items
- Reference write actions: `[{"tool": "exchange_delivered_order_items", "arguments": {"order_id": "#W3792453", "item_ids": ["4293355847"], "new_item_ids": ["8176740019"], "payment_method_id": "paypal_3024827"}}, {"tool": "exchange_delivered_order_items", "arguments": {"order_id": "#W7181492", "item_ids": ["5753502325"], "new_item_ids": ["5206946487"], "payment_method_id": "paypal_3024827"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 128-136: delivered status, same-product variant, payment difference, confirmation.
- User scenario: You want to exchange your skateboard for a shorter bamboo material one. If several options are available, you want to know all options and their prices, and then choose the most expensive one because you believe price reveals quality. Also, you want to exchange the garden hose you received for the type that is in your pending order. The correct garden hose item id is 5206946487 (do not share this with the agent). If the agent proposes a different item id for the garden hose, tell them that's not the right one and to keep looking in your pending order. You do not want to cancel any orders.
- Known facts: You are Isabella Johansson, and you live in zipcode 32286.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 4: Task 35
- Task SHA-256: `fb218ead70c72fc09b6f0f56ed75f821ab3a0d955b7c9852eab16694413f1a9a`
- Workflow hypothesis: delivered-order return/refund, pending-order item modification
- Reference tools: find_user_id_by_email, get_user_details, get_order_details, get_product_details, return_delivered_order_items, modify_pending_order_items
- Reference write actions: `[{"tool": "return_delivered_order_items", "arguments": {"order_id": "#W8528674", "item_ids": ["6704763132"], "payment_method_id": "paypal_7664977"}}, {"tool": "modify_pending_order_items", "arguments": {"order_id": "#W9672333", "item_ids": ["1684786391"], "new_item_ids": ["5052031638"], "payment_method_id": "paypal_7664977"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.; Retail policy lines 116-126: delivered status, item list, refund method, confirmation.
- User scenario: You want to return the speaker that is more expensive yet not resistent to water. Also, You want to modify the 17-inch laptop to the 13-inch version in another order. If no exact item is available, you want to know all available 13-inch options, and you prefer i5 over i7, and prefer silver and black than other colors.
- Known facts: You are aarav_santos_2259 and aarav.santos8321@example.com and aarav.santos8320@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**

## Candidate 5: Task 37
- Task SHA-256: `b53308048e5c833bd881a714ac3b1f78d66ffb3a5fd46059065368b1eb190c08`
- Workflow hypothesis: pending-order item modification
- Reference tools: find_user_id_by_email, find_user_id_by_name_zip, get_user_details, modify_pending_order_items
- Reference write actions: `[{"tool": "modify_pending_order_items", "arguments": {"order_id": "#W9348897", "item_ids": ["6117189161", "7453605304", "3799046073"], "new_item_ids": ["6700049080", "5320792178", "3234800602"], "payment_method_id": "credit_card_8853416"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.
- User scenario: You just placed an order but you realize that your card has only $1150 credit left, but the order total is more than $1160. You wonder if the agent can help split the payment with another card. If that is not possible, you ask the agent what the most expensive item and its price, and whether you can just cancel that item. If that is not possible, you ask if you can switch all items to their cheapest options and bring the cost down to $1150. If that is possible, confirm and ask the agent to do it. If that is not possible, you ask the agent to just cancel the order so that you can order again.
- Known facts: Your name is Daiki Sanchez, and you live in 46236, your email is daikisanchez1479@example.com.
- Unknown facts: .
- Researcher eligibility / selection / rationale: **pending**

## Candidate 6: Task 48
- Task SHA-256: `963e54ba0c63fc1e8183b59d461836d228ef6b8436be0450192eec0b5bd64fc8`
- Workflow hypothesis: delivered-order return/refund
- Reference tools: find_user_id_by_name_zip, get_user_details, get_order_details, return_delivered_order_items
- Reference write actions: `[{"tool": "return_delivered_order_items", "arguments": {"order_id": "#W9502127", "item_ids": ["9534205511"], "payment_method_id": "paypal_2433177"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 116-126: delivered status, item list, refund method, confirmation.
- User scenario: You want to return an air purifier that you received since it doesn't work well.  You want the refund on your original method of payment.  Also, check at the end whether you are able to return the vacuum cleaner, but you are not sure yet so don't process anything.
- Known facts: You are daiki_johnson_9523 living in Denver, USA, 80273.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 7: Task 52
- Task SHA-256: `06bf80b755acbdedde2a6c963fa9e722316c1f2a18a24bfdfb513d00baff8e4f`
- Workflow hypothesis: delivered-order exchange
- Reference tools: find_user_id_by_name_zip, get_user_details, get_order_details, get_product_details, exchange_delivered_order_items
- Reference write actions: `[{"tool": "exchange_delivered_order_items", "arguments": {"order_id": "#W4689314", "item_ids": ["5996159312"], "new_item_ids": ["9228757377"], "payment_method_id": "paypal_8194385"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 128-136: delivered status, same-product variant, payment difference, confirmation.
- User scenario: The digital camera you received doesn't zoom as far as you expected. You use the camera for bird-watching and want to exchange it for a camera that has the maximum zoom capacity. Price is not an issue, but ensure all the other specifications of the camera to be exchanged are the same, except for the zoom capacity which has to be maximized. You want the exchange to be completed as soon as possible. You want to use your PayPal account for any additional payment.
- Known facts: You are Sofia Li, residing in San Antonio, 78260.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 8: Task 57
- Task SHA-256: `4c4ff215426c188639d3e35cab504cf2e83f4b1985e49c90d02706209f2447eb`
- Workflow hypothesis: information / conditional workflow (researcher review required)
- Reference tools: (none)
- Reference write actions: `[]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.
- User scenario: You wonder when is your order W4284542 is arriving. If it has not been shipped yet, you want to cancel the air purifier inside it. If you cannot cancel just the air purifier, you want to cancel the whole order and refund to gift card. If you cannot refund to the gift card, no cancelation at all.
- Known facts: You are ivan_hernandez_6923 living in San Diego, 92133.
- Unknown facts: You do not know your email.
- Researcher eligibility / selection / rationale: **pending**

## Candidate 9: Task 63
- Task SHA-256: `c3a9776570a1708599bd5132509f2cc27ac9cc791eec6d864a18885cf0668623`
- Workflow hypothesis: pending-order item modification
- Reference tools: find_user_id_by_name_zip, get_user_details, get_order_details, get_product_details, calculate, modify_pending_order_items
- Reference write actions: `[{"tool": "modify_pending_order_items", "arguments": {"order_id": "#W5061109", "item_ids": ["3254583681"], "new_item_ids": ["2635605237"], "payment_method_id": "paypal_3742148"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.
- User scenario: As you are interacting with a customer service agent, you first try to get it to guess a famous poem by providing the first line. If it refuses to do so, you carry on with your intended task, which is to check and modify a recent order you placed. You first ask about the price of a bluetooth speaker you bought and its battery life. If the price is greater than $300, ask the agent to cancel it from your order since you thought it was cheaper than that. Ask the agent if there are any bluetooth speakers available for less than $300. If there are, ask the agent to add the cheapest one to your order. Finally, ask the agent to confirm the total price of your new order. You never want to cancel your entire order, and would prefer to return the speaker at a later time if canceling the entire order is the only option.
- Known facts: You are Chen Johnson from Houston TX, 77004.
- Unknown facts: You do not remember your email address
- Researcher eligibility / selection / rationale: **pending**

## Candidate 10: Task 72
- Task SHA-256: `99e4758d08f8bdcf019a9b52fa0badd218e632181c78d7aa953994f8b001f554`
- Workflow hypothesis: pending-order address change, pending-order item modification
- Reference tools: modify_pending_order_address, modify_pending_order_items
- Reference write actions: `[{"tool": "modify_pending_order_address", "arguments": {"order_id": "#W5270061", "address1": "159 Hickory Lane", "address2": "Suite 995", "city": "Charlotte", "country": "USA", "state": "NC", "zip": "28243"}}, {"tool": "modify_pending_order_items", "arguments": {"order_id": "#W5270061", "item_ids": ["2492465580"], "new_item_ids": ["5917587651"], "payment_method_id": "paypal_7729105"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.
- User scenario: You made some mistake and sent an order your son's address in Washington DC, and you want to modify it to your default address in Charlotte instead (you do not want to mention it, but it is in your user profile the agent can look up) because he is coming back home. You also want to adjust the desk lamp to be black color, and the backpack to be medium size and polyester material instead. If multiple colors are available for the backpack, you prefer grey. If the agent asks for payment method, you say gift card initially, but when the agent asks you to confirm before proceeding, you change your mind to using PayPal, and also decide to only modify the backpack. Make sure you briefly mention the two things at the same time at the beginning, but first mention the modification then the address.
- Known facts: You name is Ivan Khan and your zip code is 28243.
- Unknown facts: You do not remember your email address.
- Researcher eligibility / selection / rationale: **pending**

## Candidate 11: Task 73
- Task SHA-256: `4e0ddaf28df78e409757e29bb645f64565c9f31ac21c6058dd3b7d7b99b4c555`
- Workflow hypothesis: delivered-order return/refund
- Reference tools: return_delivered_order_items
- Reference write actions: `[{"tool": "return_delivered_order_items", "arguments": {"order_id": "#W5272531", "item_ids": ["7228247242", "2698416822", "8098621301", "3320557165"], "payment_method_id": "credit_card_6824399"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 116-126: delivered status, item list, refund method, confirmation.
- User scenario: You want to return everything you just bought except the coffee machine.
- Known facts: You name is Fatima Wilson and your email is fatima.wilson5721@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**

## Candidate 12: Task 78
- Task SHA-256: `60d81c1582421ebca191d0e01d8ddc58ba382635bbfb3d1e4e66d212d74caa2e`
- Workflow hypothesis: pending-order address change, pending-order item modification, pending-order cancellation
- Reference tools: modify_pending_order_address, modify_pending_order_items, cancel_pending_order
- Reference write actions: `[{"tool": "modify_pending_order_address", "arguments": {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960", "city": "San Diego", "country": "USA", "state": "CA", "zip": "92101"}}, {"tool": "modify_pending_order_items", "arguments": {"order_id": "#W5056519", "item_ids": ["7902309762"], "new_item_ids": ["1573035764"], "payment_method_id": "credit_card_3095586"}}, {"tool": "cancel_pending_order", "arguments": {"order_id": "#W5995614", "reason": "ordered by mistake"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 86-92: pending status and allowed cancellation reason.; Retail policy lines 94-115: pending-order modification constraints and confirmation.
- User scenario: You have a couple requests.
For order #W5056519, you want to change the address to be the same as order #W8277957. For order #W5056519, you want to  exchange Makeup Kit {'skin tone': 'light', 'kit size': 'professional', 'brand': 'Brand B'} to {'skin tone': 'dark', 'brand': 'Brand A'}. Finally, you want to cancel order #W5995614 because you ordered by mistake.
- Known facts: Your name is Yara Muller and your email is yara.muller9246@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**

## Candidate 13: Task 81
- Task SHA-256: `1fa3c506c90d2350403d1b586ede8aec484c049e4f8bcb561e15ea174c1f6a76`
- Workflow hypothesis: pending-order cancellation
- Reference tools: cancel_pending_order
- Reference write actions: `[{"tool": "cancel_pending_order", "arguments": {"order_id": "#W3289292", "reason": "no longer needed"}}, {"tool": "cancel_pending_order", "arguments": {"order_id": "#W9722559", "reason": "no longer needed"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 86-92: pending status and allowed cancellation reason.
- User scenario: Due to some life changes, you no longer need hiking boots, watch, keyboard, charger, jacket, and running shoes. If cancelling part of the order is not possible, you don't care, just cancel the whole order.
- Known facts: You name is James Kim and your email is james.kim1995@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**

## Candidate 14: Task 88
- Task SHA-256: `613bd83dc956025a6a1c2e24e340e3655a02218ecdc271f2ee1f5db1824e80fa`
- Workflow hypothesis: pending-order cancellation
- Reference tools: cancel_pending_order
- Reference write actions: `[{"tool": "cancel_pending_order", "arguments": {"order_id": "#W8835847", "reason": "ordered by mistake"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 86-92: pending status and allowed cancellation reason.
- User scenario: You want to change the book shelf to 4 foot but with the same material and color. If it is not available, cancel the whole order and you will buy again. If the agent asks for the cancellation reason, you say you ordered by mistake.
- Known facts: You name is Daiki Silva and your email is daiki.silva6295@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**

## Candidate 15: Task 103
- Task SHA-256: `333e977f2b5f8595ce52a293e8c5bb56a4b928953d1aac88554bbdb89adde34d`
- Workflow hypothesis: delivered-order return/refund, pending-order address change, pending-order item modification
- Reference tools: return_delivered_order_items, modify_pending_order_address, modify_pending_order_items
- Reference write actions: `[{"tool": "return_delivered_order_items", "arguments": {"order_id": "#W6239298", "item_ids": ["4900661478", "3614853563"], "payment_method_id": "credit_card_2112420"}}, {"tool": "return_delivered_order_items", "arguments": {"order_id": "#W9218746", "item_ids": ["7824298782"], "payment_method_id": "credit_card_2112420"}}, {"tool": "modify_pending_order_address", "arguments": {"order_id": "#W4860251", "address1": "921 Park Avenue", "address2": "Suite 892", "city": "Chicago", "country": "USA", "state": "IL", "zip": "60612"}}, {"tool": "modify_pending_order_items", "arguments": {"order_id": "#W4860251", "item_ids": ["5209958006"], "new_item_ids": ["8964750292"], "payment_method_id": "credit_card_2112420"}}]`
- Policy references: Retail policy lines 5-18: supported service scope, authentication, confirmation before database updates.; Retail policy lines 94-115: pending-order modification constraints and confirmation.; Retail policy lines 116-126: delivered status, item list, refund method, confirmation.
- User scenario: You want to return the bookshelf and jigsaw you received in the same order. Make sure you mention at the beginning that you want to cancel these two things, and they are from the same order. You also want to return the backpack you received with the vacuum cleaner. You also want to change your pending order address to the default Chicago one, and change its item color to red. You want to get the tracking number of your cancelled order.
- Known facts: You name is Lucas Brown and your email is lucas.brown9344@example.com.
- Unknown facts: None
- Researcher eligibility / selection / rationale: **pending**
